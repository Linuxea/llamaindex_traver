from llama_index.llms.openrouter import OpenRouter
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.tools import FunctionTool
import os


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

llm = OpenRouter(
    model="openai/gpt-4o-2024-11-20",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)

agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

while True:
    question = input("you:")
    if question == "quit":
        break
    response = agent.chat(question)
    print(response)
