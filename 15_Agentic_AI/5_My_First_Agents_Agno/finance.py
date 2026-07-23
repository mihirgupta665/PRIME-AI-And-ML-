from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()


def build_agent():
    return Agent(
        model=Groq(id="openai/gpt-oss-20b"),
        tools=[YFinanceTools(), DuckDuckGoTools()],
        markdown=True,
        add_datetime_to_context=True,
        description=(
            "You are an investment analyst that researches stock prices, "
            "analyst recommendations, and stock fundamentals."
        ),
        instructions=[
            "Use the available tools whenever needed.",
            "Format the answer in markdown.",
            "Use tables where helpful.",
        ],
        debug_mode=True,
    )


agent = build_agent()

agent.print_response(
    "Share the MSFT stock price in INR and analyst recommendation",
    show_tool_calls=True,
    stream=True,
)
