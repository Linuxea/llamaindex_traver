from llama_index.llms.openrouter import OpenRouter
import os

llm = OpenRouter(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)

print(llm.complete("讲一个笑话吧"))
