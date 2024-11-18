from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages.ai import AIMessage, ToolCall
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from src.tavily_service import get_include_domains

import streamlit as st

st.title("Software Search Helper AI")
st.write("Welcome! Ask me about any software you are looking for.")

class State(TypedDict):

    messages: Annotated[list, add_messages]

tool = TavilySearchResults(max_results=5, include_domains=get_include_domains())
tools = [tool]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()


if 'messages' not in st.session_state:
    st.session_state.messages = []

user_message = st.chat_input("Query..")

if user_message:
    st.session_state.messages.append({"role": "user", "message": user_message})
    results = []
    for event in graph.stream({"messages": [("user", user_message)]}):
        for value in event.values():
            message = value["messages"][-1]
            if type(message)  == AIMessage:
                results.append(message.content)

    st.session_state.messages.append({"role": "assistant", "message": '\n'.join(results)})


# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["message"])
    else:
        st.chat_message("assistant").markdown(message["message"])
