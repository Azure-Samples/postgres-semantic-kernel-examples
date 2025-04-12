import asyncio
import sys

import click

from .rag.cli import rag_command


@click.group()
def cli():
    """Main CLI."""
    # On Windows, set the event loop policy to WindowsSelectorEventLoopPolicy to avoid issues with asyncio
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


cli.add_command(rag_command)

if __name__ == "__main__":    
    cli()
