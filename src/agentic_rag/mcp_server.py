"""MCP server exposing the agentic RAG query tool using FastMCP."""

import logging

from fastmcp import FastMCP

from .config import load_secrets
from .agent import run_query

# Configure logging - DEBUG level for detailed agent reasoning
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Reduce noise from httpx/httpcore
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Create FastMCP server
mcp = FastMCP("agentic-rag")


@mcp.tool
async def query(question: str, max_iterations: int = 3) -> str:
    """Query the financial knowledge base using an AI agent.

    The agent will search through earnings transcripts, SEC filings,
    and other financial documents to answer your question.
    It iteratively refines searches and synthesizes information
    from multiple sources, providing citations for all information.

    Args:
        question: The question to answer about financial documents.
        max_iterations: Maximum number of LLM reasoning turns (default 3).
            Each turn can make multiple parallel tool calls.
            - 1-2: Quick, focused queries
            - 3: Default, good for most questions
            - 5+: Thorough multi-source analysis

    Returns:
        A synthesized answer with source citations.
    """
    # Load secrets before running
    settings = await load_secrets()

    # Run the agentic query
    logger.info(f"Running query: {question[:100]}... (max_iterations={max_iterations})")
    response = await run_query(
        query=question,
        settings=settings,
        max_iterations=max_iterations,
    )

    # Format response with citations
    return response.format_with_citations()


def main():
    """Entry point for the MCP server."""
    logger.info("Starting Agentic RAG MCP server...")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001, path="/mcp")


if __name__ == "__main__":
    main()
