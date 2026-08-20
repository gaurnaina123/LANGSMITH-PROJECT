import os
import requests

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
os.environ['LANGCHAIN_PROJECT'] = 'RAG AGENt'


load_dotenv()


# ---------------- Search Tool ----------------

search_tool = DuckDuckGoSearchRun()


# ---------------- Weather Tool ----------------

@tool
def get_weather_data(city: str) -> str:
    """
    Get current weather data for a given city.
    """

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    url = (
        f"https://api.weatherstack.com/current"
        f"?access_key={api_key}"
        f"&query={city}"
    )

    response = requests.get(url)

    return response.text


# ---------------- Mistral LLM ----------------

llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0
)


# ---------------- Create Agent ----------------

agent = create_agent(
    model=llm,
    tools=[
        search_tool,
        get_weather_data
    ],
    system_prompt=(
        "You are a helpful assistant. "
        "Use the search tool when you need current or "
        "up-to-date information. "
        "Use the weather tool when the user asks about weather."
    )
)


# ---------------- Run ----------------

print("Agent ready!")

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the current temperature of Gurgaon?"
        }
    ]
})


print(response["messages"][-1].content)