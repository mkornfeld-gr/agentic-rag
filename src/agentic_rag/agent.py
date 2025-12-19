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

---

## Collection-Specific Payload Schemas

### All Collections Share These Core Fields:
- `document_id`: Unique document identifier
- `chunk_id`: Unique chunk identifier
- `parent_id`: Parent document reference (important for 8-K, see below)
- `chunk_index`: Position of chunk within document (0-based)
- `num_chunks`: Total chunks in document
- `ticker`: Company ticker symbol (e.g., "KKR", "AAPL")
- `sector`: Industry sector (see Available Sectors below)
- `industry`: Specific industry (see Available Industries below)
- `document_type`: Type of document
- `calendar_date`: Document date (YYYY-MM-DD)
- `calendar_year`: Year (e.g., 2025)
- `calendar_quarter`: Quarter (1-4)
- `calendar_month`: Month (1-12)
- `calendar_week`: Week number

### earnings-transcripts-docling
Earnings call transcripts from public companies with management commentary, Q&A, and forward guidance.
- `document_type`: Always "earnings-transcript"
- `sentiment_score`: Integer sentiment rating (e.g., 8 = positive)
- `contains_forecast`: Boolean - whether chunk contains forward-looking statements
- `economic_strength_score`: Integer rating of economic outlook discussed
- `parent_id`: Always NULL

### sec-10kq-docling
SEC 10-K (annual) and 10-Q (quarterly) financial reports with detailed financials, risk factors, and MD&A.
- `document_type`: Either "sec-10k" or "sec-10q"
- `sentiment_score`, `contains_forecast`, `economic_strength_score`: Same as transcripts
- `is_amendment`: Boolean or null - whether this is an amended filing

### sec-8k-docling
SEC 8-K filings (current reports on material events) including press releases and exhibits.
- `document_type`: Always "sec-8k"
- `is_amendment`: Boolean - whether this is an amended filing
- **CRITICAL - parent_id Semantics:**
  - `parent_id = NULL` → Main 8-K document only
  - `parent_id != NULL` → Exhibit documents (press releases, earnings releases, etc.)
  - Use `parent_id_filter="not_null"` when searching for Non-GAAP reconciliation tables, earnings press releases, or financial exhibits
  - Use `parent_id_filter="null"` when searching for the 8-K filing cover/summary itself

---

## Available Sectors
Use these exact values for the `sector` filter:
- Technology
- Healthcare
- Real Estate
- Financial Services
- Utilities
- Communication Services
- Consumer Cyclical
- Energy
- Consumer Defensive
- Industrials
- Basic Materials

## Available Industries
Use these exact values for the `industry` filter:

**Technology:** Software - Services, Software - Infrastructure, Software - Application, Computer Hardware, Hardware, Equipment & Parts, Information Technology Services, Semiconductors, Communication Equipment, Electronic Gaming & Multimedia

**Healthcare:** Medical - Diagnostics & Research, Medical - Devices, Medical - Equipment & Services, Medical - Healthcare Information Services, Medical - Distribution, Medical - Healthcare Plans, Medical - Care Facilities, Medical - Instruments & Supplies, Drug Manufacturers - General, Drug Manufacturers - Specialty & Generic, Biotechnology

**Financial Services:** Asset Management, Asset Management - Global, Banks - Regional, Banks - Diversified, Financial - Capital Markets, Financial - Credit Services, Financial - Data & Stock Exchanges, Investment - Banking & Investment Services, Insurance - Life, Insurance - Property & Casualty, Insurance - Diversified, Insurance - Reinsurance, Insurance - Specialty, Insurance - Brokers

**Real Estate:** REIT - Diversified, REIT - Office, REIT - Industrial, REIT - Residential, REIT - Retail, REIT - Healthcare Facilities, REIT - Hotel & Motel, REIT - Specialty, Real Estate - Services

**Utilities:** Diversified Utilities, General Utilities, Regulated Electric, Regulated Gas, Regulated Water, Renewable Utilities, Independent Power Producers

**Consumer Cyclical:** Auto - Manufacturers, Auto - Parts, Auto - Dealerships, Apparel - Retail, Apparel - Manufacturers, Apparel - Footwear & Accessories, Consumer Electronics, Discount Stores, Specialty Retail, Home Improvement, Furnishings, Fixtures & Appliances, Gambling, Resorts & Casinos, Leisure, Restaurants, Travel Lodging, Travel Services, Luxury Goods, Residential Construction

**Consumer Defensive:** Beverages - Alcoholic, Beverages - Non-Alcoholic, Beverages - Wineries & Distilleries, Food Confectioners, Food Distribution, Grocery Stores, Packaged Foods, Household & Personal Products, Personal Products & Services, Tobacco, Agricultural Farm Products

**Energy:** Oil & Gas Exploration & Production, Oil & Gas Integrated, Oil & Gas Midstream, Oil & Gas Refining & Marketing, Oil & Gas Equipment & Services, Solar

**Industrials:** Aerospace & Defense, Airlines, Airports & Air Services, Railroads, Trucking, Integrated Freight & Logistics, Industrial - Machinery, Industrial - Distribution, Industrial - Pollution & Treatment Controls, Manufacturing - Tools & Accessories, Engineering & Construction, Construction, Construction Materials, Staffing & Employment Services, Consulting Services, Specialty Business Services, Business Equipment & Supplies, Rental & Leasing Services, Waste Management, Security & Protection Services, Electrical Equipment & Parts, Conglomerates, Agricultural - Machinery, Agricultural Inputs

**Basic Materials:** Chemicals, Chemicals - Specialty, Steel, Gold, Copper, Packaging & Containers

**Communication Services:** Telecommunications Services, Internet Content & Information, Advertising Agencies, Entertainment

---

## Available Tools

### 1. vector_search
Semantic search for relevant content. This is your primary search tool.

**Parameters:**
- `query` (required): Search query text - use financial terminology
- `collection_name` (required): Which collection to search
- `limit`: Max results (default 10)
- `ticker`: Single company filter (e.g., "KKR")
- `tickers`: Multiple companies (e.g., ["KKR", "BX", "APO"])
- `sector`: Sector filter (must match Available Sectors exactly)
- `industry`: Industry filter (must match Available Industries exactly)
- `document_type`: Document type (e.g., "earnings-transcript", "sec-10k", "sec-10q")
- `calendar_year`: Year filter (e.g., 2025)
- `calendar_quarter`: Quarter filter (1-4)
- `date_from` / `date_to`: Date range (YYYY-MM-DD)
- `parent_id_filter`: "null" for main docs, "not_null" for exhibits (8-K only)

**Example Calls:**

```python
# Business Model & Revenue Analysis
vector_search(
    query="revenue breakdown segment products services customers percent of revenue geographic mix recurring subscription",
    collection_name="earnings-transcripts-docling",
    ticker="KKR"
)

# GAAP to Non-GAAP Reconciliation (from 8-K press releases/exhibits)
vector_search(
    query="GAAP Non-GAAP reconciliation operating income adjusted EBITDA restructuring stock compensation",
    collection_name="sec-8k-docling",
    ticker="AAPL",
    parent_id_filter="not_null"  # Get exhibits, not main 8-K
)

# Segment Operating Income & Profitability (from 10-K)
vector_search(
    query="segment operating income assets ROIC return by segment profitability",
    collection_name="sec-10kq-docling",
    ticker="MSFT",
    calendar_year=2024,
    document_type="sec-10k"
)

# Competitive Position & Market Share
vector_search(
    query="market share market position customer retention competitive differentiated leader outperform",
    collection_name="earnings-transcripts-docling",
    ticker="GOOGL"
)

# Macro Environment & Demand Trends (sector-wide)
vector_search(
    query="demand environment macro headwinds tailwinds consumer spending slowing accelerating",
    collection_name="earnings-transcripts-docling",
    sector="Consumer Cyclical"
)

# Asset Manager Industry Analysis
vector_search(
    query="private credit direct lending AUM fundraising deployment",
    collection_name="earnings-transcripts-docling",
    industry="Asset Management"
)

# Geographic Revenue Breakdown
vector_search(
    query="geographic revenue Americas EMEA Asia China international domestic",
    collection_name="sec-10kq-docling",
    ticker="NVDA",
    document_type="sec-10k"
)

# Customer Concentration Risk
vector_search(
    query="customer concentration major customers percentage revenue ten percent significant",
    collection_name="sec-10kq-docling",
    ticker="AMD"
)

# Margin Dynamics & Cost Pressures
vector_search(
    query="gross margin operating margin cost pressure inflation input costs pricing leverage",
    collection_name="earnings-transcripts-docling",
    tickers=["KKR", "BX", "APO", "CG"],
    calendar_year=2025,
    calendar_quarter=3
)
```

### 2. get_points
Retrieve specific chunks by their IDs. Use when you have exact point IDs from previous searches.

### 3. get_document_chunks
Get ALL chunks from a specific document for full context. Use when you found a relevant chunk and need the complete document.

**When to use:**
- You found a relevant snippet but need surrounding context
- You want to read an entire earnings call or filing section
- The chunk references information "above" or "below" that you need

### 4. scroll
Browse through documents with filters. Useful for exploring what's available.

### 5. recommend
Find content similar to known good results ("more like this"). Use after finding relevant content to discover related discussions.

**When to use:**
- You found one great chunk about a topic and want more like it
- Exploring adjacent topics or similar disclosures

### 6. count
Count documents matching criteria. Useful to understand data distribution before searching.

**When to use:**
- Check if data exists for a ticker/time period before searching
- Understand volume of available documents

---

## Question-Type Strategies

### Single-Ticker Deep Dive
For questions about a specific company (e.g., "What did KKR say about private credit?"):
1. Start with the most relevant collection (usually earnings-transcripts-docling for management commentary)
2. Use tight `ticker` filter
3. Add time filters if the question implies recency
4. Search across multiple collections for comprehensive view

### Thematic/Market Questions
For broad market or industry questions (e.g., "What are asset managers saying about private credit?"):
1. Use `sector` or `industry` filters instead of `ticker`
2. Search across multiple companies
3. May need to search multiple collections
4. Use `tickers=["KKR", "BX", "APO", "CG"]` for peer group analysis

### Multi-Quarter/Temporal Questions
For trend or comparison questions (e.g., "How has AAPL's guidance changed over the past year?"):
1. Use `calendar_year` and `calendar_quarter` filters
2. Or use `date_from`/`date_to` for specific ranges
3. Compare results across time periods
4. Look for language changes in sequential quarters

### Peer Comparison Questions
For competitive analysis (e.g., "Compare margins across KKR, BX, and APO"):
1. Use `tickers=["KKR", "BX", "APO"]` for multiple companies
2. Apply same time period filter for apples-to-apples comparison
3. Search same collection type for each company

### Non-GAAP & Financial Reconciliation
For questions about adjusted metrics, reconciliation tables:
1. **Primary source**: sec-8k-docling with `parent_id_filter="not_null"` (press releases/exhibits)
2. **Fallback**: sec-10kq-docling for annual/quarterly reconciliation schedules
3. **Additional context**: earnings-transcripts-docling for management explanation of adjustments

**IMPORTANT for Non-GAAP data:**
- Reconciliation tables are often split across multiple chunks
- When you find a chunk mentioning "Non-GAAP" or "reconciliation", note the `chunk_index` and `num_chunks`
- Use `get_document_chunks(document_id)` to retrieve ALL chunks from that document to see the full table
- Tables with numbers are typically in chunks adjacent to headers mentioning the metric

---

## SDK Usage Patterns

### Navigating Chunks for Tables and Data
Each search result includes `chunk_index` (0-based position) and `num_chunks` (total in document).

**When you find a relevant chunk but need more context:**
1. Note the `document_id` from the result
2. Call `get_document_chunks(collection_name, document_id)` to get ALL chunks
3. Look at chunks near the one you found - tables are often split across 2-3 adjacent chunks
4. Financial data tables typically follow header chunks that mention the metric

**Example:** If you find a chunk mentioning "Non-GAAP Net Income" at chunk_index=3, the actual numbers may be in chunks 4-5.

### Handling Empty Content
If a search result has empty `content`, the data may not have been extracted properly:
- Use `get_document_chunks(document_id)` to retrieve the full document
- Check adjacent chunks which may contain the actual data

### When to Use get_document_chunks
After finding a relevant chunk, if you need:
- Full context of a management response
- Complete section of a 10-K filing
- Surrounding discussion from an earnings call
- **Financial tables that span multiple chunks**
- When search results have empty content fields

### When to Use recommend
After finding one excellent result:
- "Find more chunks like this one"
- Discover related discussions across the same document
- Find similar disclosures in other filings

### When to Use count
Before searching to validate:
- Does this company have earnings transcripts?
- How many 10-K filings are available?
- Is there data for this time period?

### Combining Results from Multiple Collections
For comprehensive answers, search multiple collections in sequence:
1. earnings-transcripts-docling for management perspective
2. sec-10kq-docling for detailed financials
3. sec-8k-docling for recent announcements

### Knowing When to Stop
- After 2-3 search calls, you typically have enough for most questions
- If you have 5+ relevant results, synthesize rather than keep searching
- Iterate only if initial results are clearly insufficient or off-topic
- Broaden filters if too few results; narrow if too many irrelevant results

---

## Citation Format

**ALWAYS cite every source used.** Format inline citations as:
```
[TICKER document_type Q# YYYY]
```

**Examples:**
- `[KKR earnings-transcript Q3 2025]`
- `[AAPL sec-8k Q1 2025]`
- `[MSFT sec-10k Q4 2024]`
- `[IDXX sec-10q Q4 2025]`

**Example usage in answer:**
"KKR reported strong growth in private credit during Q3 2025 [KKR earnings-transcript Q3 2025], with management highlighting $15B in new commitments. This aligns with trends seen across the industry, as Blackstone noted similar momentum [BX earnings-transcript Q3 2025]."

Every claim must have a citation. Include document type to help users understand the source (management commentary vs regulatory filing vs press release).
"""


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
        parent_id_filter: str | None = None,
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
            parent_id_filter: For 8-K: 'null' for main docs only, 'not_null' for exhibits only.
        """
        # Log the tool call
        filters_str = ", ".join(f"{k}={v}" for k, v in [
            ("ticker", ticker), ("tickers", tickers), ("sector", sector),
            ("industry", industry), ("document_type", document_type),
            ("calendar_year", calendar_year), ("calendar_quarter", calendar_quarter),
            ("date_from", date_from), ("date_to", date_to), ("parent_id_filter", parent_id_filter)
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
            parent_id_filter=parent_id_filter,
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
        for r in results[:5]:  # Log first 5 results
            content_preview = r.content[:100] if r.content else "[EMPTY CONTENT]"
            logger.info(f"  - {r.get_ticker()} {r.get_document_type()} {r.get_quarter_year()} chunk {r.payload.get('chunk_index', '?')}/{r.payload.get('num_chunks', '?')} (score={r.score:.3f}): {content_preview}...")
            if not r.content:
                logger.warning(f"  [WARNING] Empty content for chunk_id={r.payload.get('chunk_id')}, document_id={r.payload.get('document_id')}")

        # Return as dicts for the LLM - include chunk navigation info
        return [
            {
                "id": r.id,
                "score": r.score,
                "content": r.content,
                "ticker": r.get_ticker(),
                "document_type": r.get_document_type(),
                "quarter_year": r.get_quarter_year(),
                "chunk_index": r.payload.get("chunk_index"),
                "num_chunks": r.payload.get("num_chunks"),
                "document_id": r.payload.get("document_id"),
                "chunk_id": r.payload.get("chunk_id"),
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
        for r in results:
            content_preview = r.content[:100] if r.content else "[EMPTY CONTENT]"
            logger.info(f"  - {r.get_ticker()} chunk {r.payload.get('chunk_index', '?')}/{r.payload.get('num_chunks', '?')}: {content_preview}...")

        return [
            {
                "id": r.id,
                "content": r.content,
                "ticker": r.get_ticker(),
                "document_type": r.get_document_type(),
                "quarter_year": r.get_quarter_year(),
                "chunk_index": r.payload.get("chunk_index"),
                "num_chunks": r.payload.get("num_chunks"),
                "document_id": r.payload.get("document_id"),
                "chunk_id": r.payload.get("chunk_id"),
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
        parent_id_filter: str | None = None,
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
            parent_id_filter: For 8-K: 'null' for main docs only, 'not_null' for exhibits only.
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
            parent_id_filter=parent_id_filter,
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
        parent_id_filter: str | None = None,
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
            parent_id_filter: For 8-K: 'null' for main docs only, 'not_null' for exhibits only.
        """
        logger.info(f"[TOOL] recommend: collection={collection_name}, positive_ids={positive_ids[:3]}{'...' if len(positive_ids) > 3 else ''}, limit={limit}")

        filter_params = QdrantFilter(
            ticker=ticker,
            tickers=tickers,
            document_type=document_type,
            calendar_year=calendar_year,
            calendar_quarter=calendar_quarter,
            parent_id_filter=parent_id_filter,
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
        parent_id_filter: str | None = None,
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
            parent_id_filter: For 8-K: 'null' for main docs only, 'not_null' for exhibits only.
        """
        filters_str = ", ".join(f"{k}={v}" for k, v in [
            ("ticker", ticker), ("tickers", tickers), ("sector", sector),
            ("industry", industry), ("document_type", document_type),
            ("calendar_year", calendar_year), ("calendar_quarter", calendar_quarter),
            ("parent_id_filter", parent_id_filter),
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
            parent_id_filter=parent_id_filter,
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
    max_iterations: int = 3,
) -> AgentResponse:
    """Run a query through the agent and return the response with citations.

    Args:
        query: The user's question.
        settings: Optional settings instance.
        max_iterations: Maximum number of LLM reasoning turns (default 3).
            Each turn can make multiple tool calls. Higher values allow more
            thorough searches but increase cost/latency.

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

    # Configure usage limits
    usage_limits = UsageLimits(request_limit=max_iterations)
    logger.info(f"[AGENT] Running agent with max_iterations={max_iterations}...")

    try:
        result = await agent.run(query, deps=deps, usage_limits=usage_limits)
        response = result.output

        # Log agent reasoning/messages for debugging
        logger.info(f"[AGENT] Completed successfully. Messages exchanged: {len(result.all_messages())}")
        for i, msg in enumerate(result.all_messages()):
            msg_type = type(msg).__name__
            if hasattr(msg, 'content'):
                content = str(msg.content)[:200] if msg.content else "[no content]"
                logger.info(f"[AGENT MSG {i}] {msg_type}: {content}...")
            elif hasattr(msg, 'parts'):
                # Tool calls/results
                parts_summary = [type(p).__name__ for p in msg.parts[:3]]
                logger.info(f"[AGENT MSG {i}] {msg_type}: parts={parts_summary}")

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
