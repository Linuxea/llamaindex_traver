from llama_index.core.agent import ReActAgent
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.tools import FunctionTool
import os


class Debater:
    def __init__(self, llm=None, tools=None, topic="今天晚上吃什么"):
        self.topic = topic
        # 初始化LLM（使用OpenRouter或默认配置）
        self.llm = llm or OpenRouter(
            model="deepseek/deepseek-chat",
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        )

        # 初始化工具（继承原始agent的功能）
        self.tools = tools or [
            FunctionTool.from_defaults(
                fn=lambda a, b: a * b,
                name="multiply",
                description="Multiply two numbers",
            ),
            FunctionTool.from_defaults(
                fn=lambda a, b: a + b, name="add", description="Add two numbers"
            ),
        ]

        # 创建己方和对方两个agent
        self.self_agent = ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=False,
            max_iterations=1300,
        )
        from llama_index.core.prompts import PromptTemplate

        # Define your custom system prompt
        self.self_agent.update_prompts(
            {
                "agent_worker:system_prompt": PromptTemplate(
                    f"你作为正方辩论选手，而我作为反方辩论选手，将围绕 '{self.topic}' 展开辩论。每次发言请控制在300字以内"
                )
            }
        )

        self.opponent_agent = ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=False,
            max_iterations=1300,
        )
        self.opponent_agent.update_prompts(
            {
                "agent_worker:system_prompt": PromptTemplate(
                    f"你作为反方辩论选手，而我作为正方辩论选手，将围绕 '{self.topic}' 展开辩论。每次发言请控制在300字以内"
                )
            }
        )

    def start_debate(self, max_rounds=50):
        """启动辩论流程"""
        print(f"\n辩论主题: {self.topic}")
        history = []

        # 己方先发言（获取字符串响应）
        response = self.self_agent.chat("开始辩论").response  # 提取实际响应内容
        print(f"己方: {response}\n")
        history.append(f"己方: {response}")

        # 交替辩论
        for _ in range(max_rounds - 1):
            # 对方回应
            opponent_response = self.opponent_agent.chat(
                response
            ).response  # 同样提取响应内容
            print(f"对方: {opponent_response}\n")
            history.append(f"对方: {opponent_response}")

            input('点击请继续')
            print('\n')

            # 己方反驳
            response = self.self_agent.chat(
                opponent_response
            ).response  # 确保传递字符串
            print(f"己方: {response}\n")
            history.append(f"己方: {response}")

        return history


# 使用示例
if __name__ == "__main__":
    while True:
        topic = input("\n请输入辩论主题（输入quit退出）: ")
        if topic.lower() == "quit":
            break
        debater = Debater(topic=topic).start_debate()
