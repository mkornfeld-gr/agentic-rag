"""Pydantic models for Qdrant payloads, filters, and results."""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field, ConfigDict


class DocumentPayload(BaseModel):
    """Payload structure for documents in Qdrant.

    Allows extra fields beyond the core schema.
    """
    model_config = ConfigDict(extra="allow")

    # Core fields
    document_id: str
    chunk_id: str
    parent_id: Optional[str] = None
    chunk_index: int
    num_chunks: int

    # Company/security info
    ticker: str
    sector: str
    industry: str

    # Document metadata
    document_type: str
    calendar_date: str
    calendar_year: int
    calendar_quarter: int
    calendar_month: int
    calendar_week: int


class QdrantFilter(BaseModel):
    """Filter parameters for Qdrant queries."""
    ticker: Optional[str] = Field(None, description="Single ticker symbol (e.g., 'KKR')")
    tickers: Optional[List[str]] = Field(None, description="Multiple ticker symbols")
    sector: Optional[str] = Field(None, description="Sector name (e.g., 'Financial Services')")
    industry: Optional[str] = Field(None, description="Industry name (e.g., 'Asset Management')")
    document_type: Optional[str] = Field(None, description="Document type (e.g., 'earnings-transcript')")
    calendar_year: Optional[int] = Field(None, description="Calendar year (e.g., 2025)")
    calendar_quarter: Optional[int] = Field(None, description="Calendar quarter (1-4)")
    date_from: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class SearchResult(BaseModel):
    """A single search result from Qdrant."""
    id: str
    score: float
    content: str
    payload: Dict[str, Any]  # Flexible payload

    def get_ticker(self) -> str:
        return self.payload.get("ticker", "unknown")

    def get_document_type(self) -> str:
        return self.payload.get("document_type", "unknown")

    def get_quarter_year(self) -> str:
        q = self.payload.get("calendar_quarter", "?")
        y = self.payload.get("calendar_year", "?")
        return f"Q{q} {y}"

    def get_chunk_info(self) -> str:
        idx = self.payload.get("chunk_index", 0)
        total = self.payload.get("num_chunks", 1)
        return f"chunk {idx + 1}/{total}"

    def to_citation(self) -> str:
        """Format this result as a citation string."""
        return (
            f"{self.get_ticker()} {self.get_document_type()} "
            f"{self.get_quarter_year()} ({self.get_chunk_info()}) "
            f"[doc:{self.payload.get('document_id', 'unknown')}]"
        )


class CollectionInfo(BaseModel):
    """Information about a Qdrant collection."""
    name: str
    points_count: int
    vectors_count: int
    vector_size: int
    distance: str
    payload_schema: Dict[str, Any] = Field(default_factory=dict)


class SourceCitation(BaseModel):
    """A source citation for the final response."""
    document_id: str
    chunk_id: str
    ticker: str
    document_type: str
    calendar_quarter: int
    calendar_year: int
    chunk_index: int
    num_chunks: int
    relevance_score: float

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "SourceCitation":
        """Create a citation from a search result."""
        return cls(
            document_id=result.payload.get("document_id", ""),
            chunk_id=result.payload.get("chunk_id", ""),
            ticker=result.get_ticker(),
            document_type=result.get_document_type(),
            calendar_quarter=result.payload.get("calendar_quarter", 0),
            calendar_year=result.payload.get("calendar_year", 0),
            chunk_index=result.payload.get("chunk_index", 0),
            num_chunks=result.payload.get("num_chunks", 1),
            relevance_score=result.score,
        )

    def format(self) -> str:
        """Format as a readable citation."""
        return (
            f"{self.ticker} {self.document_type} "
            f"Q{self.calendar_quarter} {self.calendar_year} "
            f"(chunk {self.chunk_index + 1}/{self.num_chunks}) "
            f"[doc:{self.document_id}]"
        )


class AgentResponse(BaseModel):
    """Structured response from the agent."""
    answer: str = Field(description="The synthesized answer to the query")
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Sources used to generate the answer"
    )

    def format_with_citations(self) -> str:
        """Format the response with citations appended."""
        if not self.sources:
            return self.answer

        sources_text = "\n\nSources:\n"
        for src in self.sources:
            sources_text += f"- {src.format()}\n"
        return self.answer + sources_text
