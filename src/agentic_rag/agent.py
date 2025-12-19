"""Pydantic AI agent definition with Qdrant tools."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

from .config import Settings, get_settings
from .models import QdrantFilter, SearchResult, SourceCitation, AgentResponse
from .providers import get_llm_model
from . import qdrant_tools

logger = logging.getLogger(__name__)


@dataclass
class AgentDeps:
    """Dependencies injected into the agent at runtime."""
    settings: Settings
    # Track sources used during the conversation
    used_sources: List[SearchResult] = field(default_factory=list)

    def add_sources(self, results: List[SearchResult]) -> None:
        """Track sources that were used in generating the response."""
        for result in results:
            # Avoid duplicates by chunk_id
            chunk_id = result.payload.get("chunk_id")
            if chunk_id and not any(
                s.payload.get("chunk_id") == chunk_id for s in self.used_sources
            ):
                self.used_sources.append(result)


# System prompt template
SYSTEM_PROMPT = """You are a financial research assistant with access to a knowledge base of financial documents including earnings transcripts, SEC filings (10-K, 10-Q, 8-K), and other financial data.

## Available Collections
{collections_info}

## Available Tools

1. **vector_search**: Semantic search for relevant content. Requires collection_name. Use filters to narrow by:
   - ticker: Single company (e.g., "KKR")
   - tickers: Multiple companies (e.g., ["KKR", "BX", "APO"])
   - sector: Industry sector (e.g., "Financial Services")
   - industry: Specific industry (e.g., "Asset Management")
   - document_type: Type of document (e.g., "earnings-transcript")
   - calendar_year: Year (e.g., 2025)
   - calendar_quarter: Quarter (1-4)
   - date_from / date_to: Date range (YYYY-MM-DD format)
2. **get_points**: Retrieve specific chunks by their IDs
3. **get_document_chunks**: Get all chunks from a specific document for full context
4. **scroll**: Browse through documents with filters
5. **recommend**: Find content similar to known good results ("more like this")
6. **count**: Count documents matching criteria

## Your Strategy

1. **Understand the query**: Identify what information is being requested and any implied filters (company, time period, topic)
2. **Choose the right collection(s)**: Based on the query, decide which collection(s) to search:
   - Questions about earnings calls, guidance, management commentary → earnings-transcripts-docling
   - Questions about material events, announcements → sec-8k-docling
   - Questions about financials, business overview, risk factors → sec-10kq-docling
   - If unclear, search multiple collections
3. **Search strategically**: Use vector_search with appropriate filters. Be specific when the query mentions particular companies, time periods, or document types.
4. **Iterate if needed**: If initial results are insufficient:
   - Try different collections
   - Broaden or narrow filters
   - Use recommend() to find similar content to good results
   - Get full document context with get_document_chunks()
5. **Synthesize thoroughly**: Combine information from multiple sources into a comprehensive answer
6. **ALWAYS cite sources**: Every piece of information in your answer must be attributed to a specific source

## Citation Format

When you provide your final answer, you MUST cite every source used. Format inline citations as [ticker Q# YYYY] and ensure all sources are tracked for the final citation list.

Example: "KKR reported strong growth in private credit [KKR Q3 2025], while Blackstone noted similar trends [BX Q3 2025]."
"""


# ## IMPORTANT: When to Stop
#
# - After 2-3 search calls, you should have enough information to answer most questions
# - Do NOT keep searching indefinitely - synthesize what you have
# - If you have 5+ relevant results, STOP searching and provide your answer
# - Only do additional searches if initial results are clearly insufficient or off-topic
# - Aim to complete within 5-10 tool calls maximum
#


def create_agent(settings: Settings | None = None) -> Agent[AgentDeps, AgentResponse]:
    """Create and configure the RAG agent.

    Args:
        settings: Optional settings instance.

    Returns:
        Configured Pydantic AI Agent.
    """
    if settings is None:
        settings = get_settings()

    # Format system prompt with collections from config
    collections_text = settings.qdrant.get_collections_description()
    system_prompt = SYSTEM_PROMPT.format(collections_info=collections_text)

    # Create agent
    agent = Agent(
        get_llm_model(settings),
        deps_type=AgentDeps,
        output_type=AgentResponse,
        system_prompt=system_prompt,
    )

    # Register tools
    @agent.tool
    async def vector_search(
        ctx: RunContext[AgentDeps],
        query: str,
        collection_name: str,
        limit: int = 10,
        ticker: str | None = None,
        tickers: List[str] | None = None,
        sector: str | None = None,
        industry: str | None = None,
        document_type: str | None = None,
        calendar_year: int | None = None,
        calendar_quarter: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity.

        Args:
            query: The search query text.
            collection_name: The collection to search (see Available Collections).
            limit: Maximum results to return (default 10).
            ticker: Filter by single ticker symbol.
            tickers: Filter by multiple ticker symbols.
            sector: Filter by sector.
            industry: Filter by industry.
            document_type: Filter by document type (e.g., 'earnings-transcript').
            calendar_year: Filter by year.
            calendar_quarter: Filter by quarter (1-4).
            date_from: Filter by start date (YYYY-MM-DD).
            date_to: Filter by end date (YYYY-MM-DD).
        """
        # Log the tool call
        filters_str = ", ".join(f"{k}={v}" for k, v in [
            ("ticker", ticker), ("tickers", tickers), ("sector", sector),
            ("industry", industry), ("document_type", document_type),
            ("calendar_year", calendar_year), ("calendar_quarter", calendar_quarter),
            ("date_from", date_from), ("date_to", date_to)
        ] if v is not None)
        logger.info(f"[TOOL] vector_search: collection={collection_name}, query='{query[:50]}...', limit={limit}, filters=[{filters_str}]")

        filter_params = QdrantFilter(
            ticker=ticker,
            tickers=tickers,
            sector=sector,
            industry=industry,
            document_type=document_type,
            calendar_year=calendar_year,
            calendar_quarter=calendar_quarter,
            date_from=date_from,
            date_to=date_to,
        )

        results = await qdrant_tools.vector_search(
            query=query,
            collection_name=collection_name,
            limit=limit,
            filter_params=filter_params,
            settings=ctx.deps.settings,
        )

        # Track sources
        ctx.deps.add_sources(results)

        logger.info(f"[TOOL] vector_search: returned {len(results)} results")
        for r in results[:3]:  # Log first 3 results
            logger.info(f"  - {r.get_ticker()} {r.get_quarter_year()} (score={r.score:.3f}): {r.content[:80]}...")

        # Return as dicts for the LLM
        return [
            {
                "id": r.id,
                "score": r.score,
                "content": r.content,
                "ticker": r.get_ticker(),
                "document_type": r.get_document_type(),
                "quarter_year": r.get_quarter_year(),
                "chunk_info": r.get_chunk_info(),
                "document_id": r.payload.get("document_id"),
            }
            for r in results
        ]

    @agent.tool
    async def get_points(
        ctx: RunContext[AgentDeps],
        collection_name: str,
        point_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Retrieve specific chunks by their IDs.

        Args:
            collection_name: The collection to retrieve from.
            point_ids: List of point IDs to retrieve.
        """
        logger.info(f"[TOOL] get_points: collection={collection_name}, ids={point_ids[:5]}{'...' if len(point_ids) > 5 else ''}")

        results = await qdrant_tools.get_points(
            collection_name=collection_name,
            point_ids=point_ids,
        )

        ctx.deps.add_sources(results)
        logger.info(f"[TOOL] get_points: returned {len(results)} results")

        return [
            {
                "id": r.id,
                "content": r.content,
                "ticker": r.get_ticker(),
                "document_type": r.get_document_type(),
                "quarter_year": r.get_quarter_year(),
                "chunk_info": r.get_chunk_info(),
                "document_id": r.payload.get("document_id"),
            }
            for r in results
        ]

    @agent.tool
    async def get_document_chunks(
        ctx: RunContext[AgentDeps],
        collection_name: str,
        document_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all chunks belonging to a specific document for full context.

        Args:
            collection_name: The collection to retrieve from.
            document_id: The document_id to retrieve all chunks for.
        """
        logger.info(f"[TOOL] get_document_chunks: collection={collection_name}, document_id={document_id}")

        results = await qdrant_tools.get_document_chunks(
            collection_name=collection_name,
            document_id=document_id,
        )

        ctx.deps.add_sources(results)
        logger.info(f"[TOOL] get_document_chunks: returned {len(results)} chunks")

        return [
            {
                "id": r.id,
                "content": r.content,
                "chunk_index": r.payload.get("chunk_index"),
                "num_chunks": r.payload.get("num_chunks"),
                "ticker": r.get_ticker(),
                "document_type": r.get_document_type(),
                "quarter_year": r.get_quarter_year(),
            }
            for r in results
        ]

    @agent.tool
    async def scroll(
        ctx: RunContext[AgentDeps],
        collection_name: str,
        limit: int = 20,
        offset: str | None = None,
        ticker: str | None = None,
        tickers: List[str] | None = None,
        sector: str | None = None,
        industry: str | None = None,
        document_type: str | None = None,
        calendar_year: int | None = None,
        calendar_quarter: int | None = None,
    ) -> Dict[str, Any]:
        """Browse through documents with optional filtering.

        Args:
            collection_name: The collection to browse.
            limit: Number of results per page.
            offset: Pagination offset from previous call.
            ticker: Filter by single ticker symbol.
            tickers: Filter by multiple ticker symbols.
            sector: Filter by sector.
            industry: Filter by industry.
            document_type: Filter by document type.
            calendar_year: Filter by year.
            calendar_quarter: Filter by quarter.
        """
        logger.info(f"[TOOL] scroll: collection={collection_name}, limit={limit}, offset={offset}")

        filter_params = QdrantFilter(
            ticker=ticker,
            tickers=tickers,
            sector=sector,
            industry=industry,
            document_type=document_type,
            calendar_year=calendar_year,
            calendar_quarter=calendar_quarter,
        )

        results, next_offset = await qdrant_tools.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            filter_params=filter_params,
        )

        logger.info(f"[TOOL] scroll: returned {len(results)} results, next_offset={next_offset}")

        return {
            "results": [
                {
                    "id": r.id,
                    "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                    "ticker": r.get_ticker(),
                    "document_type": r.get_document_type(),
                    "quarter_year": r.get_quarter_year(),
                }
                for r in results
            ],
            "next_offset": next_offset,
        }

    @agent.tool
    async def recommend(
        ctx: RunContext[AgentDeps],
        collection_name: str,
        positive_ids: List[str],
        negative_ids: List[str] | None = None,
        limit: int = 10,
        ticker: str | None = None,
        tickers: List[str] | None = None,
        document_type: str | None = None,
        calendar_year: int | None = None,
        calendar_quarter: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Find content similar to positive examples (and dissimilar to negative ones).

        Use this for "more like this" exploration after finding good results.

        Args:
            collection_name: The collection to search.
            positive_ids: Point IDs to find similar results to.
            negative_ids: Optional point IDs to find dissimilar results from.
            limit: Maximum results to return.
            ticker: Filter by single ticker symbol.
            tickers: Filter by multiple ticker symbols.
            document_type: Filter by document type.
            calendar_year: Filter by year.
            calendar_quarter: Filter by quarter.
        """
        logger.info(f"[TOOL] recommend: collection={collection_name}, positive_ids={positive_ids[:3]}{'...' if len(positive_ids) > 3 else ''}, limit={limit}")

        filter_params = QdrantFilter(
            ticker=ticker,
            tickers=tickers,
            document_type=document_type,
            calendar_year=calendar_year,
            calendar_quarter=calendar_quarter,
        )

        results = await qdrant_tools.recommend(
            collection_name=collection_name,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            limit=limit,
            filter_params=filter_params,
        )

        ctx.deps.add_sources(results)
        logger.info(f"[TOOL] recommend: returned {len(results)} results")

        return [
            {
                "id": r.id,
                "score": r.score,
                "content": r.content,
                "ticker": r.get_ticker(),
                "document_type": r.get_document_type(),
                "quarter_year": r.get_quarter_year(),
                "chunk_info": r.get_chunk_info(),
                "document_id": r.payload.get("document_id"),
            }
            for r in results
        ]

    @agent.tool
    async def count(
        ctx: RunContext[AgentDeps],
        collection_name: str,
        ticker: str | None = None,
        tickers: List[str] | None = None,
        sector: str | None = None,
        industry: str | None = None,
        document_type: str | None = None,
        calendar_year: int | None = None,
        calendar_quarter: int | None = None,
    ) -> int:
        """Count documents matching the specified filters.

        Useful for understanding data distribution before searching.

        Args:
            collection_name: The collection to count in.
            ticker: Filter by single ticker symbol.
            tickers: Filter by multiple ticker symbols.
            sector: Filter by sector.
            industry: Filter by industry.
            document_type: Filter by document type.
            calendar_year: Filter by year.
            calendar_quarter: Filter by quarter.
        """
        filters_str = ", ".join(f"{k}={v}" for k, v in [
            ("ticker", ticker), ("tickers", tickers), ("sector", sector),
            ("industry", industry), ("document_type", document_type),
            ("calendar_year", calendar_year), ("calendar_quarter", calendar_quarter),
        ] if v is not None)
        logger.info(f"[TOOL] count: collection={collection_name}, filters=[{filters_str}]")

        filter_params = QdrantFilter(
            ticker=ticker,
            tickers=tickers,
            sector=sector,
            industry=industry,
            document_type=document_type,
            calendar_year=calendar_year,
            calendar_quarter=calendar_quarter,
        )

        result = await qdrant_tools.count(
            collection_name=collection_name,
            filter_params=filter_params,
        )

        logger.info(f"[TOOL] count: returned {result}")
        return result

    return agent


async def run_query(
    query: str,
    settings: Settings | None = None,
) -> AgentResponse:
    """Run a query through the agent and return the response with citations.

    Args:
        query: The user's question.
        settings: Optional settings instance.

    Returns:
        AgentResponse with answer and source citations.
    """
    logger.info(f"[AGENT] Starting query: {query[:100]}{'...' if len(query) > 100 else ''}")

    if settings is None:
        settings = get_settings()

    # Create agent
    agent = create_agent(settings)
    logger.info(f"[AGENT] Using model: {settings.llm.provider}/{settings.llm.model}")

    # Create dependencies
    deps = AgentDeps(settings=settings)

    # Run agent (limits commented out for haiku - it's fast/cheap enough)
    # usage_limits = UsageLimits(
    #     request_limit=15,      # Max LLM requests
    #     tool_calls_limit=20,   # Max tool executions
    # )
    logger.info("[AGENT] Running agent...")

    try:
        result = await agent.run(query, deps=deps)
        response = result.output
    except UsageLimitExceeded as e:
        # Hit the limit - synthesize response from what we gathered
        logger.warning(f"[AGENT] Usage limit exceeded: {e}. Synthesizing partial response.")

        # Build sources from what was collected
        sources = [
            SourceCitation.from_search_result(src)
            for src in deps.used_sources
        ]

        # Create a partial response with suggestions
        collected_tickers = list(set(s.ticker for s in sources if s.ticker))
        collected_types = list(set(s.document_type for s in sources if s.document_type))

        partial_answer = (
            "**Note: Search limit reached. Here's what I found so far:**\n\n"
            f"I gathered {len(sources)} relevant document chunks from the knowledge base. "
            f"The sources cover: {', '.join(collected_tickers[:10]) if collected_tickers else 'various companies'}.\n\n"
            "**Suggested areas for follow-up queries:**\n"
        )

        # Suggest narrower follow-up queries
        if len(collected_tickers) > 3:
            partial_answer += f"- Focus on specific companies (e.g., one of: {', '.join(collected_tickers[:5])})\n"
        if len(collected_types) > 1:
            partial_answer += f"- Narrow to specific document types: {', '.join(collected_types)}\n"
        partial_answer += "- Add time filters (specific quarter/year)\n"
        partial_answer += "- Ask about a specific topic or metric\n"

        response = AgentResponse(answer=partial_answer, sources=sources)

    # Ensure sources are populated from tracked sources
    if not response.sources and deps.used_sources:
        response.sources = [
            SourceCitation.from_search_result(src)
            for src in deps.used_sources
        ]

    logger.info(f"[AGENT] Completed. Sources used: {len(response.sources)}, Answer length: {len(response.answer)} chars")

    return response
