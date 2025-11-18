from dotenv import load_dotenv

load_dotenv()

# imports
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from schemas import AgentResponse

# Define tools and model
tools = [TavilySearch()]
llm = ChatOllama(model="mistral")

# Create agent with structured output using ToolStrategy
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=ToolStrategy(AgentResponse)
)


def main():
    # Prepare the message for the agent (new API expects 'messages' key)
    user_query = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_query}]
    })
    # If using structured output, access it via result["structured_response"]
    print(result.get("structured_response", result))

if __name__ == "__main__":
    main()
