from agno.agent import Agent    
from agno.models.groq import Groq
from agno.models.google import GeminiInteractions

from dotenv import load_dotenv

from agno.db.sqlite import SqliteDb

load_dotenv()

db = SqliteDb


def build_agent():
    return Agent(
        model=Groq(id="qwen/qwen3.6-27b"),
        # model=GeminiInteractions(id="gemini-3.5-flash"),
        markdown=True,
        # instructions="You are a helpfull, well experienced and expert travel agent.",
    )


agent = build_agent()

# agent.print_response("My budget is 1L INR, should I travel to Goa or Phuket?")
agent.print_response("what happend today at delhi and jantar mantar")
