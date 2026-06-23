import json
import requests
from openai import OpenAI


class GaodeAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, city: str, extensions: str) -> dict:
        url = "https://restapi.amap.com/v3/weather/weatherInfo"
        params = {
            "city": city,
            "key": self.api_key,
            "extensions": extensions,   # base=实况, all=预报
            "output": "JSON",
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "1":
            raise RuntimeError(
                f"Amap API error: {data.get('info')} ({data.get('infocode')})"
            )
        return data

    def get_current_weather(self, city: str) -> dict:
        return self._request(city, "base")

    def get_weather_forecast(self, city: str) -> dict:
        return self._request(city, "all")


client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

AMAP_KEY = ""


def get_current_weather(city: str) -> dict:
    api = GaodeAPI(AMAP_KEY)
    return api.get_current_weather(city)


def get_weather_forecast(city: str) -> dict:
    api = GaodeAPI(AMAP_KEY)
    return api.get_weather_forecast(city)


TOOLS = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current real-time weather conditions of a city. Use this ONLY for questions about the exact weather right now (e.g., current temperature, current wind, current humidity",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, for example: 北京, 上海, Shenzhen"
                }
            },
            "required": ["city"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_weather_forecast",
        "description": "Get the multi-day weather forecast of a city, including today's overall summary and upcoming days. Use this for questions about high/low temperatures, day/night weather, tomorrow, and future days.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, for example: 北京, 上海, Shenzhen"
                }
            },
            "required": ["city"],
            "additionalProperties": False
        },
        "strict": True
    }
]


def run_tool(tool_name: str, arguments_json: str):
    args = json.loads(arguments_json)
    city = args["city"]

    if tool_name == "get_current_weather":
        raw_res = get_current_weather(city)
        # 只保留 lives 里的关键信息
        if "lives" in raw_res:
            return raw_res["lives"][0] 
        return raw_res

    if tool_name == "get_weather_forecast":
        raw_res = get_weather_forecast(city)
        # 只给模型看关键的 casts 列表
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


def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input.strip():
            continue
        response = client.responses.create(
            model="Qwen3.5-4B-AWQ-4bit",
            input=user_input,
            tools=TOOLS,
            tool_choice="auto",
        )

        while True:
            function_calls = [item for item in response.output if item.type == "function_call"]

            if not function_calls:
                print("模型最终回答：")
                print(response.output_text)
                break

            tool_outputs = []
            for call in function_calls:
                print(f"模型调用工具: {call.name}")
                print(f"工具参数: {call.arguments}")

                result = run_tool(call.name, call.arguments)

                # 避免把敏感字段打出来
                safe_result = json.loads(json.dumps(result, ensure_ascii=False))
                for key in ("key", "sec_code", "sec_code_debug"):
                    safe_result.pop(key, None)

                print("工具返回：")
                print(json.dumps(safe_result, ensure_ascii=False, indent=2))

                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

            response = client.responses.create(
                model="Qwen3.5-4B-AWQ-4bit",
                input=[
                    {"role": "user", "content": user_input},
                    *response.output,
                    *tool_outputs,
                ],
                tools=TOOLS,
                tool_choice="auto",
            )


if __name__ == "__main__":
    main()