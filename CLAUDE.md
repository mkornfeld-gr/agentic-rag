# CLAUDE.md

Development guide for the Agentic RAG MCP server.

## Project Structure

```
src/agentic_rag/
├── __init__.py
├── config.py        # Settings, Prefect secret loading
├── models.py        # Pydantic models (filters, results, citations)
├── providers.py     # LLM and embedding providers
├── qdrant_tools.py  # Qdrant client wrapper and operations
├── agent.py         # Pydantic AI agent with tools + SYSTEM_PROMPT
└── mcp_server.py    # FastMCP server entry point
```

## Key Files

### `config.py`
- `QdrantSettings`: Host, port, collections dict
- `LLMSettings`: Provider, model, API key
- `EmbeddingSettings`: Model, API key
- `load_secrets()`: Async function that loads API keys from Prefect secret blocks

### `models.py`
- `QdrantFilter`: Filter parameters (ticker, sector, date range, parent_id_filter, etc.)
- `SearchResult`: Single search result with helper methods
- `SourceCitation`: Citation for final response
- `AgentResponse`: Structured output with answer + sources

### `providers.py`
- `get_llm_model()`: Returns Pydantic AI model based on provider setting
- `generate_embedding()`: Generates embeddings via OpenAI

### `qdrant_tools.py`
- `get_qdrant_client()`: Singleton async client
- `build_qdrant_filter()`: Converts QdrantFilter to Qdrant Filter object (supports IsNullCondition for parent_id)
- `vector_search()`: Semantic search
- `get_points()`: Retrieve by IDs
- `get_document_chunks()`: All chunks for a document_id
- `scroll()`: Paginated browsing
- `recommend()`: Similar content via RecommendQuery
- `count()`: Count matching documents

### `agent.py`
- `AgentDeps`: Dependencies dataclass with settings and used_sources tracking
- `SYSTEM_PROMPT`: Comprehensive instructions (~300 lines) including:
  - Collection-specific payload schemas
  - Available sectors (11) and industries (100+)
  - Example tool calls
  - Question-type strategies
  - Chunk navigation guidance for tables
- `create_agent()`: Creates Pydantic AI agent with all tools registered
- `run_query(query, settings, max_iterations)`: Main entry point with configurable iteration limit

### `mcp_server.py`
- Creates FastMCP server with single `query` tool
- Exposes `max_iterations` parameter (default 3)
- DEBUG-level logging for agent reasoning

## Filter Fields

The `QdrantFilter` model supports:
- `ticker`: Single ticker symbol
- `tickers`: Multiple ticker symbols (OR)
- `sector`: Sector name (must match exactly)
- `industry`: Industry name (must match exactly)
- `document_type`: e.g., "earnings-transcript", "sec-10k", "sec-10q"
- `calendar_year`: Year filter
- `calendar_quarter`: Quarter (1-4)
- `date_from` / `date_to`: Date range (YYYY-MM-DD)
- `parent_id_filter`: **"null"** for main docs, **"not_null"** for exhibits (8-K only)

## The parent_id_filter (8-K Documents)

SEC 8-K filings have a special structure:
- **Main 8-K document** (`parent_id = NULL`): The filing cover/summary
- **Exhibit documents** (`parent_id != NULL`): Press releases, earnings releases, financial exhibits

For Non-GAAP reconciliation tables, use:
```python
vector_search(
    query="GAAP Non-GAAP reconciliation...",
    collection_name="sec-8k-docling",
    ticker="META",
    parent_id_filter="not_null"  # Get exhibits, not main 8-K
)
```

## Adding a New Collection

1. Update `config.py`:
```python
collections: dict[str, str] = Field(default_factory=lambda: {
    "earnings-transcripts-docling": "Earnings call transcripts...",
    "sec-8k-docling": "SEC 8-K filings...",
    "sec-10kq-docling": "SEC 10-K/10-Q reports...",
    "new-collection-docling": "Description of new collection",  # Add here
})
```

2. Update `SYSTEM_PROMPT` in `agent.py` with collection-specific payload fields

3. Rebuild:
```bash
docker compose down && docker compose build --no-cache && docker compose up -d
```

## Adding a New Filter Field

1. Update `models.py` QdrantFilter:
```python
class QdrantFilter(BaseModel):
    # ... existing fields
    new_field: Optional[str] = Field(None, description="Description")
```

2. Update `qdrant_tools.py` build_qdrant_filter():
```python
if filter_params.new_field:
    conditions.append(
        FieldCondition(key="new_field", match=MatchValue(value=filter_params.new_field))
    )
```

3. Update tool signatures in `agent.py` to include the new parameter.

4. Update `SYSTEM_PROMPT` in `agent.py` to document the new filter.

## Changing the LLM

Edit `config.py`:
```python
class LLMSettings(BaseModel):
    provider: Literal["openai", "anthropic", "ollama"] = "anthropic"
    model: str = "claude-sonnet-4-5"  # Change model here
```

Supported providers:
- `anthropic`: Claude models (claude-sonnet-4-5, claude-opus-4-5, etc.)
- `openai`: GPT models (gpt-4o, gpt-4-turbo, etc.)
- `ollama`: Local models (requires base_url)

## Adjusting Iteration Limits

The `max_iterations` parameter controls LLM reasoning turns (not tool calls):
- Each turn can make multiple parallel tool calls
- Default: 3 turns (think → search → refine → answer)
- For complex queries requiring more searches: increase to 5-7

```python
# MCP tool call
query(question="...", max_iterations=5)
```

In code:
```python
response = await run_query(
    query="...",
    settings=settings,
    max_iterations=5,
)
```

## Debugging

View agent logs:
```bash
docker logs -f agentic-rag
```

Log output shows:
- `[AGENT] Starting query: ...` - Query received
- `[AGENT] Using model: ...` - Which LLM
- `[AGENT] Running agent with max_iterations=N...` - Iteration limit
- `[TOOL] vector_search: ...` - Tool calls with all parameters
- `[TOOL] vector_search: returned N results` - Result count
- `  - TICKER doc_type Q# YYYY chunk X/Y (score=0.XXX): content...` - Each result preview
- `  [WARNING] Empty content for chunk_id=...` - Empty content detection
- `[AGENT MSG N] MessageType: content...` - Agent reasoning messages
- `[AGENT] Completed. Sources used: N` - Final stats

## Common Issues

### `UsageLimitExceeded`
Agent hit iteration limit. Increase `max_iterations` or narrow your query.

### Empty content in search results
The data extraction may have failed for some documents. The agent is instructed to use `get_document_chunks()` as a fallback.

### Non-GAAP data not found
Make sure to use `parent_id_filter="not_null"` for 8-K filings - reconciliation tables are in exhibits, not main docs.

### `'AsyncQdrantClient' object has no attribute 'recommend'`
Qdrant client API changed. Use `query_points()` with `RecommendQuery` instead.

### Point ID format errors
Collection uses integer IDs. The `_parse_point_id()` function handles conversion.

### API key errors
Ensure Prefect secrets exist:
- `openai-api-key`
- `anthropic-api-key`

And container has `PREFECT_API_URL` set.

## Testing Locally

```bash
# Start server
uv run python -m agentic_rag.mcp_server

# Test with curl (SSE)
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/call", "params": {"name": "query", "arguments": {"question": "What is KKR?", "max_iterations": 3}}}'
```

## Rebuild Cycle

```bash
docker compose down && docker compose build --no-cache && docker compose up -d
```

Quick check:
```bash
docker logs agentic-rag 2>&1 | tail -20
```
