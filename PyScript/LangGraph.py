import json
import requests
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

class GaodeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, city: str, extensions: str) -> dict:
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {
            "city": city,
            "key": self.api_key,
            "extensions": extensions,
            "output": "JSON",
        }
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

# 使用 @tool 装饰器，LangChain 会自动从函数签名和 docstring 中提取 JSON Schema
@tool
def get_current_weather(city: str) -> dict:
    """获取指定城市的当前实时天气情况。用于询问当前的温度、风力、湿度等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_current_weather(city)
    if "lives" in raw_res:
        return raw_res["lives"][0]
    return raw_res

@tool
def get_weather_forecast(city: str) -> dict:
    """获取指定城市的多日天气预报。用于询问今天整体、明天或未来的天气、最高/最低温等。"""
    api = GaodeAPI(AMAP_KEY)
    raw_res = api.get_weather_forecast(city)
    if "forecasts" in raw_res:
        forecast = raw_res["forecasts"][0]
        return {
            "city": forecast["city"],
            "casts": [
                {k: v for k, v in c.items() if "float" not in k}
                for c in forecast["casts"]
            ]
        }
    return raw_res

tools = [get_current_weather, get_weather_forecast]

llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    model="Qwen3.5-4B-AWQ-4bit"
)
llm_with_tools = llm.bind_tools(tools)

# 定义一个包含 messages 列表的 State (直接使用内置的 MessagesState)
def agent_node(state: MessagesState):
    """大模型节点：将历史消息传递给 LLM，并返回新的 AIMessage"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 初始化图构建器
builder = StateGraph(MessagesState)

# 添加节点
builder.add_node("agent", agent_node)
# 创建一个工具执行节点 (使用内置的 ToolNode，它会自动处理 tool_calls)
builder.add_node("tools", ToolNode(tools))

# 定义图的边
builder.add_edge(START, "agent")

# 设置条件边 (Conditional Edge)
# tools_condition 是 LangGraph 内置的路由：
# 如果 agent 节点返回了 tool_calls -> 流转到 "tools"
# 如果没有 -> 流转到 END
builder.add_conditional_edges("agent", tools_condition)

# 工具执行完毕后，返回给大模型继续思考
builder.add_edge("tools", "agent")

# 人类确认机制
# 使用 MemorySaver 持久化状态图，以便我们可以在特定节点前暂停它
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # 在流转到 "tools" 节点前打断点
)


def main():
    # 配置线程 ID，LangGraph 依赖这个来区分不同的对话状态
    thread_config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input.strip():
            continue

        # 向图中推入用户的消息并开始流转
        events = graph.stream(
            {"messages": [("user", user_input)]}, 
            thread_config, 
            stream_mode="values"
        )
        
        # 打印初始的流转过程 (通常是 Agent 给出的回答或 Tool call 的请求)
        for event in events:
            last_message = event["messages"][-1]
            # 简单过滤，避免把用户的输入再打印一遍
            if last_message.type == "ai" and last_message.content:
                print(f"Agent: {last_message.content}")

        # 检查图是否在断点处暂停
        state = graph.get_state(thread_config)
        
        # 如果 state.next 包含 "tools"，说明它被我们设置的 interrupt_before 拦截了
        while state.next and state.next[0] == "tools":
            last_message = state.values["messages"][-1]
            tool_calls = last_message.tool_calls
            
            tool_names = [tc["name"] for tc in tool_calls]
            print(f"\n[拦截] 大模型请求执行工具: {tool_names}")
            print(f"工具参数: {json.dumps(tool_calls[0]['args'], ensure_ascii=False)}")
            
            # 提示终端用户确认
            decision = input("是否允许执行？(Y/N): ")
            
            if decision.lower() == 'y':
                print("授权通过，继续执行...")   
                # 传入 None 并使用同样的 thread_config，图会从断点处继续往下流转 (执行 tools -> 回到 agent)
                events = graph.stream(None, thread_config, stream_mode="values")
                for event in events:
                    last_msg = event["messages"][-1]
                    if last_msg.type == "ai" and last_msg.content:
                        print(f"Agent: {last_msg.content}")
                
                # 刷新状态，看是否还有后续动作
                state = graph.get_state(thread_config)
            
            else:
                print("已拒绝执行该工具。")
                # 如果拒绝，必须向状态中手动注入一条假装是 Tool 返回的拒绝消息。
                # 否则大模型会卡在等待工具返回的状态里。
                denial_messages = [
                    ToolMessage(
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        content="用户拒绝了该工具的执行。系统警告：你现在没有任何真实的天气数据，严禁利用自己的知识库猜测或编造天气情况！请直接向用户说明没有数据。"
                    )
                    for tc in tool_calls
                ]
                
                # 将拒绝消息更新到状态图中，假装是 tools 节点执行的结果，并直接跳回到 agent 节点
                graph.update_state(thread_config, {"messages": denial_messages}, as_node="tools")
                
                # 再次 stream，让 agent 根据“被拒绝”这个事实继续说话
                events = graph.stream(None, thread_config, stream_mode="values")
                for event in events:
                    last_msg = event["messages"][-1]
                    if last_msg.type == "ai" and last_msg.content:
                        print(f"Agent: {last_msg.content}")
                
                break # 拒绝后跳出当前确认循环

if __name__ == "__main__":
    main()