from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import GeminiInteractions
from agno.team import Team


from agno.tools.duckduckgo import DuckDuckGoTools


from dotenv import load_dotenv

load_dotenv()

eng_agent = Agent(name="English Agent", role="You answer questions in English")
ger_agent = Agent(name="German Agent", role="You answer questions in German")
hindi_agent = Agent(name="Hindi Agent", role="You answer questions in Hindi")

team_leader = Team(
    name="Answer and Translation Team ",
    members=[eng_agent, ger_agent, hindi_agent],
    model=Groq(id="qwen/qwen3.6-27b"),
    markdown=True,
    show_members_responses=True,
    instructions=
    """
        All members agents must responds ot answer the query in thier specific language.
        Do not route to just on agent.
        Output the response of all agents.
    """
)


team_leader.print_response(
    "What is the capital of India?",
    show_tool_calls=True,
)
