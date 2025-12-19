"""LLM and embedding provider abstraction."""

from typing import List

import openai
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from .config import get_settings, Settings


def get_llm_model(settings: Settings | None = None) -> Model:
    """Get the configured LLM model for Pydantic AI.

    Args:
        settings: Optional settings instance. If not provided, uses global settings.

    Returns:
        A Pydantic AI Model instance configured for the selected provider.
    """
    if settings is None:
        settings = get_settings()

    llm = settings.llm

    if llm.provider == "anthropic":
        provider = AnthropicProvider(api_key=llm.api_key)
        return AnthropicModel(llm.model, provider=provider)
    elif llm.provider == "openai":
        provider = OpenAIProvider(api_key=llm.api_key, base_url=llm.base_url)
        return OpenAIModel(llm.model, provider=provider)
    elif llm.provider == "ollama":
        # Ollama uses OpenAI-compatible API
        provider = OpenAIProvider(
            api_key="ollama",
            base_url=llm.base_url or "http://localhost:11434/v1",
        )
        return OpenAIModel(llm.model, provider=provider)
    else:
        raise ValueError(f"Unknown LLM provider: {llm.provider}")


def get_embedding_client(settings: Settings | None = None) -> openai.AsyncOpenAI:
    """Get an async OpenAI client for embeddings.

    Args:
        settings: Optional settings instance. If not provided, uses global settings.

    Returns:
        An AsyncOpenAI client configured for embeddings.
    """
    if settings is None:
        settings = get_settings()

    embed = settings.embedding

    return openai.AsyncOpenAI(
        api_key=embed.api_key,
        base_url=embed.base_url,
    )


async def generate_embedding(
    text: str,
    settings: Settings | None = None,
) -> List[float]:
    """Generate an embedding vector for the given text.

    Used to convert search queries into vectors for Qdrant similarity search.

    Args:
        text: The text to embed (typically a search query).
        settings: Optional settings instance.

    Returns:
        A list of floats representing the embedding vector.
    """
    if settings is None:
        settings = get_settings()

    client = get_embedding_client(settings)

    response = await client.embeddings.create(
        model=settings.embedding.model,
        input=text,
    )

    return response.data[0].embedding
