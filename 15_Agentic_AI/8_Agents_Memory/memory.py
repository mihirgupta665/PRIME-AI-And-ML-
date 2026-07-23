import os
from typing import Iterable

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.groq import Groq
from agno.models.fallback import FallbackConfig
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
from rich.pretty import pprint


load_dotenv()


class SafeDuckDuckGoTools(DuckDuckGoTools):
    def search_news(self, query: str, max_results: int = 5) -> str:
        try:
            return super().search_news(query=query, max_results=max_results)
        except Exception as exc:
            try:
                web_results = super().web_search(query=query, max_results=max_results)
                return (
                    "News-specific results were unavailable, so general web results were used instead.\n"
                    f"Reason: {exc}\n"
                    f"Fallback results:\n{web_results}"
                )
            except Exception as fallback_exc:
                return (
                    "No live search results were available for this query.\n"
                    f"News error: {exc}\n"
                    f"Web fallback error: {fallback_exc}"
                )


db = SqliteDb(db_file="agno.db")
db.clear_memories()


def build_model(model_id: str) -> Groq:
    return Groq(
        id=model_id,
        max_retries=2,
        timeout=30,
        temperature=0.2,
        max_tokens=400,
    )


def build_agent() -> Agent:
    return Agent(
        db=db,
        model=build_model("llama-3.3-70b-versatile"),
        fallback_config=FallbackConfig(
            on_rate_limit=[
                build_model("llama-3.1-8b-instant"),
            ],
            on_error=[
                build_model("llama-3.3-70b-versatile"),
            ],
        ),
        tools=[
            SafeDuckDuckGoTools(
                backend="duckduckgo",
                region="in-en",
                timelimit="d",
                timeout=15,
            )
        ],
        instructions=[
            "If a news search returns no result, try general web search context instead of stopping.",
            "If live information is unavailable, say that clearly instead of returning an empty response.",
            "Keep answers concise and mention uncertainty when facts are still developing.",
        ],
        markdown=True,
        add_history_to_context=True,
        enable_user_memories=True,
    )


def ask(agent: Agent, prompts: Iterable[str], user_id: str) -> None:
    for prompt in prompts:
        print(f"\nUSER: {prompt}")
        try:
            agent.print_response(prompt, user_id=user_id)
        except Exception as exc:
            print(f"ASSISTANT ERROR: {exc}")


agent = build_agent()
user_id = "mihir@gmail.com"

ask(
    agent,
    prompts=[
        "What happend today at delhi and jantar mantar",
        "was there any casualties reported till now",
    ],
    user_id=user_id,
)

memories = agent.get_user_memories(user_id=user_id)

print("\nMEMORIES:")
pprint(memories)
