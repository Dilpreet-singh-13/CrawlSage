import os
from typing import List
from dotenv import load_dotenv
from dataclasses import dataclass

from pydantic_ai import RunContext, Agent
from pydantic_ai.models.gemini import GeminiModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from db_setup import SitePage
from crawler import get_embedding

SYSTEM_PROMPT = """
You are an expert at Crawl4ai - a blazing-fast, AI-ready web crawling tailored for large language models, AI agents, and data pipelines, that you have access to all the documentation to, including examples, an API reference, and other resources.

Your only job is to assist with this or help with code/logic using anything from what crawl4ai offers and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

load_dotenv()


@dataclass
class RAGDeps:
    async_sessionmaker: sessionmaker


model = GeminiModel(os.getenv("GEMINI_MODEL"), api_key=os.getenv("GEMINI_API_KEY"))
rag_agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
    deps_type=RAGDeps,
)


@rag_agent.tool
async def retrieve_doc_chunk(ctx: RunContext[RAGDeps], user_query: str) -> str:
    """
    Retrive relevent documentation chunks based on the user query with RAG.
    Generates the embeddings for the user query and retrives relevent documentation chunk based on cosine distance from the embeddings of the chunks.

    Args:
        ctx: The context incuding the sqlalchemy session used for database access
        user_query: The user's query or question. Used to find the relevent documentation chunks

    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """

    try:
        # Get embedding for the user query
        user_query_embeddings = await get_embedding(user_query)

        # Query database (default: supabase) using the ORM
        # Uses cosine similarity between the embedding vector of the user query and doc chunks
        async with ctx.deps.async_sessionmaker() as session:
            query = (
                select(
                    SitePage.url,
                    SitePage.chunk_number,
                    SitePage.title,
                    SitePage.content,
                )
                .order_by(SitePage.embedding.cosine_distance(user_query_embeddings))
                .limit(5)
            )
            result = await session.execute(query)
            chunks = result.fetchall()
            if not chunks:
                return "No relevant documentation found."

        # Format the result
        formatted_chunks = []
        for url, chunk_number, title, content in chunks:
            chunk_text = f"""# {title}\nURL: {url} | Chunk number for this URL: {chunk_number}\n\n{content}"""
            formatted_chunks.append(chunk_text)

        # Join all chunks
        return "\n\n----\n\n".join(formatted_chunks)
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return "An error occurred while retrieving documentation."


@rag_agent.tool
async def list_documentation_urls(ctx: RunContext[RAGDeps]) -> List[str]:
    """
    Retrives a list of all available Crawl4ai documentation pages that have been crawled.

    Args:
        ctx: The context incuding the sqlalchemy session used for database access

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        async with ctx.deps.async_sessionmaker() as session:
            urls = await session.scalars(select(SitePage.url).distinct())
            url_list = list(urls)
            if not url_list:
                print("Error retrieving documentation pages.")
                return []

            return url_list
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


@rag_agent.tool
async def get_page_context(ctx: RunContext[RAGDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        async with ctx.deps.async_sessionmaker() as session:
            query = (
                select(SitePage.title, SitePage.content)
                .where(SitePage.url == url)
                .order_by(SitePage.chunk_number)
            )
            result = await session.execute(query)
            chunks = result.fetchall()
            if not chunks:
                print(f"No content found for URL: {url}")
                return f"No content found for URL: {url}"

        # Format and return the structured content
        return "\n\n".join(f"## {title}\n\n{content}" for title, content in chunks)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return "An error occurred while retrieving page content."
