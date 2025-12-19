# Agentic RAG

An agentic RAG (Retrieval-Augmented Generation) system that connects to Qdrant vector database and exposes a single `query` tool via MCP (Model Context Protocol). The agent autonomously searches financial documents, iterates to find relevant information, and synthesizes comprehensive answers with citations.

## Features

- **Agentic search**: The AI agent autonomously decides which collections to search, what filters to apply, and when to dig deeper
- **Multiple collections**: Searches across earnings transcripts, SEC 8-K filings, and 10-K/10-Q reports
- **Smart filtering**: Filter by ticker, sector, industry, document type, date range, quarter, year, and parent_id (for 8-K exhibits)
- **Citation tracking**: All answers include source citations with document type, ticker, quarter/year
- **Chunk navigation**: Agent understands document structure and can retrieve full documents when tables span multiple chunks
- **Configurable depth**: Control search thoroughness via `max_iterations` parameter
- **Recommendation API**: "Find more like this" functionality using Qdrant's recommendation engine
- **MCP exposure**: Single `query` tool exposed via FastMCP with streamable HTTP transport

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│   FastMCP       │────▶│  Pydantic AI    │
│  (Claude, etc)  │     │   Server        │     │     Agent       │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        │                                ▼                                │
                        │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
                        │  │vector_search │  │  get_points  │  │   recommend  │  Tools   │
                        │  └──────────────┘  └──────────────┘  └──────────────┘          │
                        │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
                        │  │    scroll    │  │    count     │  │get_doc_chunks│          │
                        │  └──────────────┘  └──────────────┘  └──────────────┘          │
                        └────────────────────────────────┬────────────────────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │     Qdrant      │
                                               │  Vector Store   │
                                               └─────────────────┘
```

## Collections

The agent has access to these Qdrant collections:

| Collection | Description | Key Fields |
|------------|-------------|------------|
| `earnings-transcripts-docling` | Earnings call transcripts from public companies | `sentiment_score`, `contains_forecast`, `economic_strength_score` |
| `sec-8k-docling` | SEC 8-K filings (current reports on material events) | `is_amendment`, `parent_id` (NULL=main doc, NOT NULL=exhibits) |
| `sec-10kq-docling` | SEC 10-K (annual) and 10-Q (quarterly) financial reports | `is_amendment`, `document_type` (sec-10k or sec-10q) |

### 8-K Document Structure

SEC 8-K filings have exhibits (press releases, earnings releases) as separate documents:
- `parent_id = NULL`: Main 8-K filing
- `parent_id != NULL`: Exhibit documents (where Non-GAAP reconciliation tables live)

Use `parent_id_filter="not_null"` to search only exhibits.

## Quick Start

### Docker (Recommended)

```bash
docker compose up --build
```

The MCP server will be available at `http://localhost:8001/mcp`

### Local Development

```bash
# Install dependencies
uv sync

# Run the server
uv run python -m agentic_rag.mcp_server
```

## Configuration

Configuration is managed via `src/agentic_rag/config.py`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `qdrant.host` | `10.10.150.104` | Qdrant server host |
| `qdrant.port` | `6333` | Qdrant server port |
| `llm.provider` | `anthropic` | LLM provider (anthropic, openai, ollama) |
| `llm.model` | `claude-sonnet-4-5` | Model to use |
| `embedding.model` | `text-embedding-3-small` | OpenAI embedding model |

### API Keys

API keys are loaded from Prefect secret blocks:
- `openai-api-key` - For embeddings
- `anthropic-api-key` - For Claude models

The container needs access to Prefect API:
```yaml
environment:
  - PREFECT_API_URL=http://10.10.150.104:4200/api
```

## MCP Integration

### Claude Desktop / Claude Code

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "agentic-rag": {
      "type": "streamable-http",
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

### Available Tools

The MCP server exposes a single tool:

#### `query`

Query the financial knowledge base using an AI agent.

**Parameters:**
- `question` (string, required): The question to answer about financial documents
- `max_iterations` (int, default 3): Maximum LLM reasoning turns. Each turn can make multiple tool calls.
  - 1-2: Quick, focused queries
  - 3: Default, good for most questions
  - 5+: Thorough multi-source analysis (e.g., Non-GAAP reconciliation)

**Returns:** A synthesized answer with source citations

**Example:**
```
Question: "What did KKR say about private credit in their Q3 2025 earnings call?"

Answer: KKR reported strong momentum in private credit during Q3 2025...

Sources:
- KKR earnings-transcript Q3 2025 (chunk 12/45) [doc:abc123]
- KKR earnings-transcript Q3 2025 (chunk 15/45) [doc:abc123]
```

**Example with more iterations:**
```
Question: "Get Meta's Q4 2025 Non-GAAP EPS and reconciliation table"
max_iterations: 7

Answer: Meta reported Non-GAAP diluted EPS of $X.XX for Q4 2025...
[Full reconciliation table data]

Sources:
- META sec-8k Q4 2025 (chunk 3/6) [doc:xyz789]
```

## Agent Tools

The agent has access to these internal tools:

| Tool | Description |
|------|-------------|
| `vector_search` | Semantic search with filters (ticker, sector, industry, date range, parent_id_filter) |
| `get_points` | Retrieve specific chunks by ID |
| `get_document_chunks` | Get all chunks from a document (for tables spanning multiple chunks) |
| `scroll` | Browse documents with filters |
| `recommend` | Find similar content ("more like this") |
| `count` | Count documents matching criteria |

## Filter Options

All filter-supporting tools accept these parameters:

| Filter | Description | Example |
|--------|-------------|---------|
| `ticker` | Single company | `"KKR"` |
| `tickers` | Multiple companies | `["KKR", "BX", "APO"]` |
| `sector` | Industry sector | `"Financial Services"` |
| `industry` | Specific industry | `"Asset Management"` |
| `document_type` | Document type | `"earnings-transcript"`, `"sec-10k"`, `"sec-10q"` |
| `calendar_year` | Year | `2025` |
| `calendar_quarter` | Quarter | `3` |
| `date_from` / `date_to` | Date range | `"2025-01-01"` |
| `parent_id_filter` | 8-K only: main vs exhibits | `"null"` or `"not_null"` |

## Payload Schema

Documents in Qdrant have this payload structure:

```json
{
  "document_id": "uuid",
  "chunk_id": "uuid",
  "parent_id": null,
  "chunk_index": 0,
  "num_chunks": 45,
  "ticker": "KKR",
  "sector": "Financial Services",
  "industry": "Asset Management",
  "document_type": "earnings-transcript",
  "calendar_date": "2025-07-31",
  "calendar_year": 2025,
  "calendar_quarter": 3,
  "calendar_month": 7,
  "calendar_week": 31,
  "sentiment_score": 8,
  "contains_forecast": true,
  "economic_strength_score": 6
}
```

## Available Sectors

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

## Development

```bash
# Install dev dependencies
uv sync

# Run locally
uv run python -m agentic_rag.mcp_server

# View logs (DEBUG level shows agent reasoning)
docker logs -f agentic-rag
```

## Debugging

The server runs with DEBUG-level logging. Key log patterns:

```
[AGENT] Starting query: ...
[AGENT] Running agent with max_iterations=N...
[TOOL] vector_search: collection=X, query='...', filters=[...]
  - TICKER doc_type Q# YYYY chunk X/Y (score=0.XXX): content preview...
  [WARNING] Empty content for chunk_id=...  # Data extraction issue
[AGENT MSG N] ModelRequest: ...  # Agent reasoning
[AGENT] Completed. Sources used: N
```

## License

MIT
