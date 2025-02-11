import asyncio

import click

from .arxiv_chat_example import chat_with_arxiv_papers, load_arxiv_papers, search_arxiv_papers


@click.group(name="rag")
def rag_command():
    """RAG commands."""
    pass


@rag_command.command()
@click.option("-n", "--num-papers", default=100)
@click.option("-t", "--topic", default="RAG")
@click.option("-c", "--category", default="cs.AI") 
@click.option("-e", "--env-file-path", default=".env")
def load(num_papers: int = 100, topic: str = "RAG", category: str = "cs.AI", env_file_path: str = ".env"):
    """Load data."""
    asyncio.run(
        load_arxiv_papers(total_papers=num_papers, topic=topic, category=category, env_file_path=env_file_path)
    )


@rag_command.command()
@click.argument("query")
@click.option("-n", "--count", default=5)
def search(query: str, count: int):
    """Load data."""
    asyncio.run(search_arxiv_papers(query, count))


@rag_command.command()
def chat():
    """Chat with the model."""
    asyncio.run(chat_with_arxiv_papers())
