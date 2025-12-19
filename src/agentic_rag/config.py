"""Configuration management using Pydantic settings and Prefect secrets."""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field


class QdrantSettings(BaseModel):
    """Qdrant vector database configuration."""
    host: str = "10.10.150.104"
    port: int = 6333
    vector_size: int = 1536
    distance_metric: Literal["cosine", "euclid", "dot"] = "cosine"

    # Available collections with descriptions for the agent
    collections: dict[str, str] = Field(default_factory=lambda: {
        "earnings-transcripts-docling": "Earnings call transcripts from public companies",
        "sec-8k-docling": "SEC 8-K filings (current reports on material events)",
        "sec-10kq-docling": "SEC 10-K (annual) and 10-Q (quarterly) financial reports",
    })

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def get_collection_names(self) -> list[str]:
        """Get list of available collection names."""
        return list(self.collections.keys())

    def get_collections_description(self) -> str:
        """Get formatted description of collections for system prompt."""
        lines = []
        for name, description in self.collections.items():
            lines.append(f"- **{name}**: {description}")
        return "\n".join(lines)


class LLMSettings(BaseModel):
    """LLM provider configuration."""
    provider: Literal["openai", "anthropic", "ollama"] = "anthropic"
    model: str = "claude-sonnet-4-5"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    embedding_dim: int = 1536


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    return Settings()


async def load_secrets() -> Settings:
    """Load Prefect secrets into settings. Call this in async context before using API keys."""
    from prefect.blocks.system import Secret

    settings = get_settings()

    # Load OpenAI API key for embeddings
    if settings.embedding.api_key is None:
        secret = await Secret.load('openai-api-key')
        settings.embedding.api_key = secret.get()

    # Load LLM API key based on provider
    if settings.llm.api_key is None:
        if settings.llm.provider == "anthropic":
            secret = await Secret.load('anthropic-api-key')
            settings.llm.api_key = secret.get()
        elif settings.llm.provider == "openai":
            secret = await Secret.load('openai-api-key')
            settings.llm.api_key = secret.get()
        # ollama doesn't need an API key

    return settings
