import os
import asyncio
import json
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict

from crawl4ai import AsyncWebCrawler, CacheMode, RateLimiter, SemaphoreDispatcher
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.dialects.postgresql import insert
from google import genai
from google.genai import types
from pydantic import BaseModel

from db_setup import SitePage


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    embedding: List[float]


async_session_maker = None


def initialize_session_maker():
    """
    Initialize a global async SQLAlchemy sessionmaker. Gets credentials from environment variables.
    """
    global async_session_maker

    # Only create the sessionmaker if it hasn't been created yet
    if async_session_maker is None:
        load_dotenv()

        USER = os.getenv("user")
        PASSWORD = os.getenv("password")
        HOST = os.getenv("host")
        PORT = os.getenv("port")
        DBNAME = os.getenv("dbname")

        DATABASE_URL = f"postgresql+asyncpg://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"

        async_engine = create_async_engine(
            DATABASE_URL,
            pool_pre_ping=True,  # Ensures connections are valid before using them
            pool_size=5,  # We only run 5 concurrent crawlers
            max_overflow=5,  # Allow up to 5 connections beyond pool_size
        )

        async_session_maker = sessionmaker(bind=async_engine, class_=AsyncSession)

    return async_session_maker


async def generate_chunks(text: str, chunk_size: int = 4096) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif "\n\n" in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind("\n\n")
            if (
                last_break > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif ". " in chunk:
            # Find the last sentence break
            last_period = chunk.rfind(". ")
            if (
                last_period > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks


class ResponseSchema(BaseModel):
    title: str
    summary: str


async def get_title_and_summary(text, url) -> Dict[str, str]:
    """Extract the title and summary for the given text using GPT-4 mini (defalt)"""

    DEV_PROMPT = """You are an AI that extracts titles and summaries from documentation chunks.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative.

    Return a JSON object with 'title' and 'summary' keys. Use the JSON schema given below.
    {
        'title': str,
        'summary': str
    }
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            config=types.GenerateContentConfig(
                system_instruction=DEV_PROMPT,
                response_mime_type="application/json",
                response_schema=ResponseSchema,
            ),
            contents=[f"URL: {url}\n\nCONTENT:\n{text}"],
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"status": "failed", "message": "Error getting title and summary"}


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        response = client.models.embed_content(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
            contents=[text],
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Error getting embeddings: {e} ")
        return [0.0] * 768


async def process_chunk(
    chunk: str, chunk_number: int, url: str
) -> ProcessedChunk | None:
    """
    Process a single chunk of text.
    Get the embeddings, title and summary for the chunk.

    Return: Instance of "ProcessedChunk" class
    """
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)

    if isinstance(extracted, dict) and extracted.get("status") == "failed":
        print(f"Failed to extract title/summary for chunk {chunk_number}")
        return None

    title = extracted.get("title", f"Chunk {chunk_number} for URL: {url}")
    summary = extracted.get("summary", "No summary available")

    # Get embedding
    embedding = await get_embedding(chunk)

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=title,
        summary=summary,
        content=chunk,
        embedding=embedding,
    )


async def store_chunk(chunk: ProcessedChunk) -> None:
    """Store a processed chunk into Supabase database."""

    async_session_maker = initialize_session_maker()
    # insert the chunk, if it exists then update its contents
    statement = (
        insert(SitePage)
        .values(
            url=chunk.url,
            chunk_number=chunk.chunk_number,
            title=chunk.title,
            summary=chunk.summary,
            content=chunk.content,
            embedding=chunk.embedding,
        )
        .on_conflict_do_update(
            index_elements=["url", "chunk_number"],
            set_={
                "title": chunk.title,
                "summary": chunk.summary,
                "content": chunk.content,
                "embedding": chunk.embedding,
            },
        )
    )

    async with async_session_maker() as session:
        try:
            await session.execute(statement)
            await session.commit()
        except Exception as e:
            await session.rollback()
            print(f"Error inserting chunk: {e}")


async def process_and_store_docs(url: str, markdown: str):
    """
    Split the provided markdown into chunks for efficient storage and use.
    Also stores the chunks in the database in parallel.
    """
    # Split markdown into chunks
    chunks = await generate_chunks(markdown)

    # Process chunks in parallel
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)

    # Store chunks in parallel
    insert_tasks = [
        store_chunk(chunk) for chunk in processed_chunks if chunk is not None
    ]
    await asyncio.gather(*insert_tasks)


async def crawl_parallel(url_list: List[str], max_concurrency: int = 5):
    """
    Crawl multiple URLs in parallel with a concurrency limit.
    Default concurrency limit = 5
    Uses a small delay between each crawl and has 3 (default) retries.
    """
    EXTRA_ARGS = [
        "--disable-gpu",  # No GPU rendering needed in headless mode
        "--disable-dev-shm-usage",  # Avoid issues in low-memory environments (especially Docker)
        "--no-sandbox",  # For headless mode, allows restricted environments to run
        "--disable-software-rasterizer",  # Disable software rendering to improve performance
        "--disable-background-timer-throttling",  # Prevents Chrome from throttling background tabs
        "--disable-backgrounding-occluded-windows",  # Prevents Chrome from pausing inactive pages
        "--disable-renderer-backgrounding",  # Prevents deprioritizing background pages
        "--disable-extensions",  # Disables unnecessary browser extensions
        "--disable-features=IsolateOrigins,site-per-process",  # Reduces isolation, improving performance
    ]

    browser_config = BrowserConfig(headless=True, verbose=False, extra_args=EXTRA_ARGS)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["nav", "footer"],
        markdown_generator=DefaultMarkdownGenerator(),
    )

    # There is a 0-3 second delay between each crawl and maximum of 20 second delay
    rate_limit = RateLimiter(
        base_delay=(0, 3), max_delay=20.0, max_retries=3, rate_limit_codes=[429, 503]
    )

    semaphore_dispatcher = SemaphoreDispatcher(
        max_session_permit=max_concurrency, rate_limiter=rate_limit
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Crawl all URLs in parallel
        results = await crawler.arun_many(
            urls=url_list, config=run_config, dispatcher=semaphore_dispatcher
        )

        for res in results:
            if res.success:
                print(f"Successfully crawled {res.url}")
                await process_and_store_docs(res.url, res.markdown)
            else:
                print(f"Failed: {res.url} - {res.error_message}")


async def geneate_crawl4ai_sitemap(
    link: str = "https://docs.crawl4ai.com/",
) -> list[str]:
    """
    Generates the sitemap for the crawl4ai docs.
    Default Docs Url: https://docs.crawl4ai.com/

    Returns: list of URLs
    """

    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        exclude_external_links=True,
        remove_overlay_elements=True,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=link, run_config=run_config)

        if result.success:
            # According to the structure of the docs, we only need to extract the "internal" urls
            # i.e the ones that share the same domain
            sitemap_links = [
                link.get("href", "") for link in result.links.get("internal", [])
            ]
            return sitemap_links
        else:
            print(f"Failed to crawl {link}: {result.error_message}")
            print(f"Status code: {result.status_code}")
            return []


async def main():
    url_list = await geneate_crawl4ai_sitemap()

    if not url_list:
        print("No URLs found to crawl")
        return

    # temporary: remove some URLs that aren't useful to crawl
    REMOVE_URLS = [
        "https://docs.crawl4ai.com/#how-you-can-support",
        "https://old.docs.crawl4ai.com",
    ]
    url_list = [url for url in url_list if url not in REMOVE_URLS]

    print(f"Crawling {len(url_list)} URLs.")
    await crawl_parallel(url_list)


if __name__ == "__main__":
    asyncio.run(main())
