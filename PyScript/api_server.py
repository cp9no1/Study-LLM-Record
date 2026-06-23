import json
import requests
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, Field, ConfigDict, field_validator
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 1. 基础接口定义与 Pydantic 校验
class Message(BaseModel):
    role: str = Field(..., description="角色，仅允许 user 或 assistant")
    content: str = Field(..., description="消息内容")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v == "system":
            raise ValueError("出于安全考虑，禁止客户端直接传递 'system' 角色。")
        if v not in ["user", "assistant"]:
            raise ValueError("角色必须是 'user' 或 'assistant'。")
        return v

class ChatRequest(BaseModel):
    # 严格模式：拒绝任何未在模型中定义的额外字段
    model_config = ConfigDict(extra="forbid")
    
    question: str = Field(..., min_length=1, max_length=1000, description="用户的当前提问")
    history: List[Message] = Field(default_factory=list, max_length=50, description="历史对话上下文")
    max_tokens: Optional[int] = Field(default=1024, le=4096, description="最大生成 token 数量限制")
    thread_id: str = Field(default="default_thread", description="用于 LangGraph 记忆的会话 ID")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    thread_id: str = Field(..., description="当前会话的 ID")
    usage: dict = Field(default_factory=dict, description="Token 消耗情况（预留）")

# 2. 天气工具与大模型配置
class GaodeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, city: str, extensions: str) -> dict:
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {"city": city, "key": self.api_key, "extensions": extensions, "output": "JSON"}
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "1":
            raise RuntimeError(f"Amap API error: {data.get('info')} ({data.get('infocode')})")
        return data

    def get_current_weather(self, city: str) -> dict:
        return self._request(city, "base")

    def get_weather_forecast(self, city: str) -> dict:
        return self._request(city, "all")

AMAP_KEY = ""

@tool
def get_current_weather(city: str) -> dict:
    """获取指定城市的当前实时天气情况。用于询问当前的温度、风力、湿度等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_current_weather(city)
    return raw_res["lives"][0] if "lives" in raw_res else raw_res

@tool
def get_weather_forecast(city: str) -> dict:
    """获取指定城市的多日天气预报。返回的数据包含今天整体、明天或未来的天气、最高/最低温等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_weather_forecast(city)
    if "forecasts" in raw_res:
        forecast = raw_res["forecasts"][0]
        return {
            "city": forecast["city"],
            "casts": [{k: v for k, v in c.items() if "float" not in k} for c in forecast["casts"]]
        }
    return raw_res

tools = [get_current_weather, get_weather_forecast]

llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    model="Qwen3.5-4B-AWQ-4bit"
)
llm_with_tools = llm.bind_tools(tools)

# 3. LangGraph 状态图构建 (移除终端中断机制)
def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 4. FastAPI 路由与中间件拦截

app = FastAPI(title="LangGraph Agent API", version="1.0.0")

# 模拟的鉴权白名单
VALID_API_KEYS = {"my-secret-token-2026"}

def verify_headers(
    x_api_key: str = Header(..., description="API 鉴权 Key"),
    content_type: str = Header(..., description="必须为 application/json")
):
    """验证 Header：Content-Type 与 API Key"""
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_headers)])
async def chat_endpoint(request: ChatRequest):
    """
    接收用户提问，调用 LangGraph 智能体处理，并返回结果。
    """
    # 1. 组装 LangGraph 需要的 message 格式
    formatted_messages = []
    
    # 注入历史记录
    for msg in request.history:
        formatted_messages.append((msg.role, msg.content))
        
    # 注入当前问题
    formatted_messages.append(("user", request.question))

    # 2. 配置 Thread ID 以维持状态图记忆
    thread_config = {"configurable": {"thread_id": request.thread_id}}

    try:
        # 3. 调用 Graph (同步调用，等待图完全执行完毕)
        # 注意：此处使用 invoke 而不是 stream，是为了直接获取最终状态
        final_state = graph.invoke(
            {"messages": formatted_messages}, 
            config=thread_config
        )
        
        # 4. 获取 Agent 的最后一条回复
        final_message = final_state["messages"][-1]
        
        return ChatResponse(
            answer=final_message.content,
            thread_id=request.thread_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 执行出错: {str(e)}")

if __name__ == "__main__":
    # 使用 Uvicorn 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8080)