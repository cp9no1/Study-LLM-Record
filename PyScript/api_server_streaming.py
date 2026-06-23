import json
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
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
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=1, max_length=1000)
    history: List[Message] = Field(default_factory=list, max_length=50)
    max_tokens: Optional[int] = Field(default=1024, le=4096)
    thread_id: str = Field(default="default_thread")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的最终回答")
    thread_id: str = Field(..., description="当前会话的 ID")

# 2. 工具与大模型配置
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
    model="Qwen3.5-4B-AWQ-4bit",
    streaming=True # 开启底层模型的流式支持
)
llm_with_tools = llm.bind_tools(tools)

# 3. LangGraph 状态图构建
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

# 4. FastAPI 路由与 SSE 流式返回
app = FastAPI(title="LangGraph Agent API", version="1.0.0")

VALID_API_KEYS = {"my-secret-token-2026"}

def verify_headers(x_api_key: str = Header(...), content_type: str = Header(...)):
    if content_type != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_headers)])
async def chat_endpoint(request: ChatRequest):
    formatted_messages = [(msg.role, msg.content) for msg in request.history]
    formatted_messages.append(("user", request.question))
    thread_config = {"configurable": {"thread_id": request.thread_id}}
    final_state = await graph.ainvoke({"messages": formatted_messages}, config=thread_config)
    return ChatResponse(answer=final_state["messages"][-1].content, thread_id=request.thread_id)

# 流式输出 SSE 接口
@app.post("/chat/stream", dependencies=[Depends(verify_headers)])
async def chat_stream_endpoint(request: ChatRequest):
    # 组装消息列表
    formatted_messages = [(msg.role, msg.content) for msg in request.history]
    formatted_messages.append(("user", request.question))
    thread_config = {"configurable": {"thread_id": request.thread_id}}

    # 定义异步生成器函数
    async def event_generator():
        try:
            # astream 配合 stream_mode="messages" 能捕获到底层大模型的每一次消息增量 (Chunk)
            async for msg, metadata in graph.astream(
                {"messages": formatted_messages},
                config=thread_config,
                stream_mode="messages",
            ):
                # 条件 1: 必须是 agent 节点发出的消息 (排除了 tools 节点的消息)
                # 条件 2: 必须有 content 文本 (排除了 agent 请求调用工具时的内部不可见 json)
                if metadata.get("langgraph_node") == "agent" and msg.content:
                    
                    # 组装我们要发给前端的 JSON 数据块
                    chunk_data = json.dumps({"token": msg.content}, ensure_ascii=False)
                    
                    # 严格遵守 SSE 协议格式：以 "data: " 开头，以两个换行符 "\n\n" 结尾
                    yield f"data: {chunk_data}\n\n"

            # 模型生成完毕后，按照业界惯例发送 [DONE] 标识
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    # 使用 StreamingResponse 返回生成器，并指定媒体类型为 text/event-stream
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5530)