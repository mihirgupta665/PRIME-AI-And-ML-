from agno.agent import Agent    
from agno.models.groq import Groq
from agno.models.google import GeminiInteractions

from dotenv import load_dotenv

from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()


def build_agent():
    return Agent(
        # model=Groq(id="qwen/qwen3.6-27b"),
        model=GeminiInteractions(id="gemini-3.5-flash"),
        tools=[DuckDuckGoTools()],
        markdown=True,
        # instructions="You are a helpfull, well experienced and expert travel agent.",
        instructions="You are a helpfull, expert  agent.",
        add_datetime_to_context=True,
    )


agent = build_agent()

# agent.print_response("My budget is 1L INR, should I travel to Goa or Phuket?")
agent.print_response(
    "what happend today at delhi and jantar mantar", show_tool_calls=True
)
