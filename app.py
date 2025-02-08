from dotenv import load_dotenv
from typing import List

import gradio as gr
from pydantic_ai.messages import ModelResponse, ModelRequest, TextPart, UserPromptPart

from crawler import initialize_session_maker
from crawl4ai_RAG_agent import RAGDeps, rag_agent

load_dotenv()
async_session_maker = initialize_session_maker()

chat_history = []


async def run_agent(user_input: str, history: List):
    """
    Run the agent, keeps mesage history.
    """
    # add user input to history
    history.append({"role": "user", "content": user_input})

    # prepare dependencies and run agent
    deps = RAGDeps(async_sessionmaker=async_session_maker)
    result = await rag_agent.run(
        user_prompt=user_input, deps=deps, message_history=chat_history
    )

    # Add new messages from this run to the chat history for pydanticAI agent
    filtered_messages = [msg for msg in result.new_messages()]
    chat_history.extend(filtered_messages)

    # add model response to chat histroy for pydanticAI agent
    chat_history.append(ModelResponse(parts=[TextPart(content=result.data)]))

    # add model response to history
    history.append({"role": "assistant", "content": result.data})

    yield result.data


demo = gr.ChatInterface(fn=run_agent, type="messages", title="Crawl4ai Agent")
demo.launch()
