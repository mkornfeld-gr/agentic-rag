"""Qdrant client wrapper and tool implementations."""

from typing import List, Optional, Dict, Any, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    ScoredPoint,
    PointStruct,
    RecommendQuery,
    RecommendInput,
    IsNullCondition,
    PayloadField,
)

from .config import get_settings, Settings
from .models import QdrantFilter, SearchResult, CollectionInfo
from .providers import generate_embedding


# Global client instance
_client: Optional[AsyncQdrantClient] = None


async def get_qdrant_client(settings: Settings | None = None) -> AsyncQdrantClient:
    """Get or create the Qdrant client singleton."""
    global _client
    if _client is None:
        if settings is None:
            settings = get_settings()
        _client = AsyncQdrantClient(
            url=settings.qdrant.url,
            timeout=60,
        )
    return _client


def build_qdrant_filter(filter_params: QdrantFilter | None) -> Optional[Filter]:
    """Convert QdrantFilter to Qdrant's native Filter format.

    Args:
        filter_params: Our filter model with optional filter criteria.

    Returns:
        A Qdrant Filter object, or None if no filters specified.
    """
    if filter_params is None:
        return None

    conditions = []

    # Single ticker
    if filter_params.ticker:
        conditions.append(FieldCondition(
            key="ticker",
            match=MatchValue(value=filter_params.ticker)
        ))

    # Multiple tickers (OR condition)
    if filter_params.tickers:
        conditions.append(FieldCondition(
            key="ticker",
            match=MatchAny(any=filter_params.tickers)
        ))

    # Sector
    if filter_params.sector:
        conditions.append(FieldCondition(
            key="sector",
            match=MatchValue(value=filter_params.sector)
        ))

    # Industry
    if filter_params.industry:
        conditions.append(FieldCondition(
            key="industry",
            match=MatchValue(value=filter_params.industry)
        ))

    # Document type
    if filter_params.document_type:
        conditions.append(FieldCondition(
            key="document_type",
            match=MatchValue(value=filter_params.document_type)
        ))

    # Calendar year
    if filter_params.calendar_year:
        conditions.append(FieldCondition(
            key="calendar_year",
            match=MatchValue(value=filter_params.calendar_year)
        ))

    # Calendar quarter
    if filter_params.calendar_quarter:
        conditions.append(FieldCondition(
            key="calendar_quarter",
            match=MatchValue(value=filter_params.calendar_quarter)
        ))

    # Date range
    if filter_params.date_from or filter_params.date_to:
        range_params = {}
        if filter_params.date_from:
            range_params["gte"] = filter_params.date_from
        if filter_params.date_to:
            range_params["lte"] = filter_params.date_to
        conditions.append(FieldCondition(
            key="calendar_date",
            range=Range(**range_params)
        ))

    # Parent ID filter (for 8-K main docs vs exhibits)
    # "null" = main docs only, "not_null" = exhibits only (e.g., press releases)
    # Note: IsNullCondition is a separate condition type, not part of FieldCondition
    # We'll handle this by returning a filter with must/must_not as appropriate
    # For now, we add to a separate list and handle below
    must_not_conditions = []
    if filter_params.parent_id_filter == "null":
        # parent_id IS NULL - use IsNullCondition in must
        conditions.append(IsNullCondition(
            is_null=PayloadField(key="parent_id")
        ))
    elif filter_params.parent_id_filter == "not_null":
        # parent_id IS NOT NULL - use IsNullCondition in must_not
        must_not_conditions.append(IsNullCondition(
            is_null=PayloadField(key="parent_id")
        ))

    if not conditions and not must_not_conditions:
        return None

    return Filter(
        must=conditions if conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )


def _scored_point_to_result(point: ScoredPoint) -> SearchResult:
    """Convert a Qdrant ScoredPoint to our SearchResult model."""
    payload = point.payload or {}
    return SearchResult(
        id=str(point.id),
        score=point.score,
        content=payload.get("text", payload.get("content", "")),
        payload=payload,
    )


# =============================================================================
# Tool implementations
# =============================================================================

async def list_collections() -> List[str]:
    """List all available collections in Qdrant.

    Returns:
        List of collection names.
    """
    client = await get_qdrant_client()
    collections = await client.get_collections()
    return [c.name for c in collections.collections]


async def get_collection_info(collection_name: str) -> CollectionInfo:
    """Get detailed information about a collection.

    Args:
        collection_name: Name of the collection.

    Returns:
        CollectionInfo with metadata about the collection.
    """
    client = await get_qdrant_client()
    info = await client.get_collection(collection_name)

    # Extract vector size and distance
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict):
        # Named vectors
        first_vector = next(iter(vectors_config.values()))
        vector_size = first_vector.size
        distance = str(first_vector.distance)
    else:
        # Single vector config
        vector_size = vectors_config.size
        distance = str(vectors_config.distance)

    return CollectionInfo(
        name=collection_name,
        points_count=info.points_count or 0,
        vectors_count=info.vectors_count or 0,
        vector_size=vector_size,
        distance=distance,
        payload_schema=info.payload_schema or {},
    )


async def vector_search(
    query: str,
    collection_name: str,
    limit: int = 10,
    filter_params: QdrantFilter | None = None,
    settings: Settings | None = None,
) -> List[SearchResult]:
    """Search for similar documents using vector similarity.

    Args:
        query: The search query text.
        collection_name: Name of the collection to search.
        limit: Maximum number of results to return.
        filter_params: Optional filters to apply.
        settings: Optional settings instance.

    Returns:
        List of SearchResult objects with matching documents.
    """
    client = await get_qdrant_client()

    # Embed the query
    query_vector = await generate_embedding(query, settings)

    # Build filter
    qdrant_filter = build_qdrant_filter(filter_params)

    # Search
    results = await client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [_scored_point_to_result(point) for point in results.points]


def _parse_point_id(point_id: str) -> int | str:
    """Parse a point ID string to the appropriate type (int or UUID string)."""
    try:
        return int(point_id)
    except ValueError:
        # If it can't be parsed as int, assume it's a UUID string
        return point_id


async def get_points(
    collection_name: str,
    point_ids: List[str],
) -> List[SearchResult]:
    """Retrieve specific points by their IDs.

    Args:
        collection_name: Name of the collection.
        point_ids: List of point IDs to retrieve.

    Returns:
        List of SearchResult objects for the requested points.
    """
    client = await get_qdrant_client()

    # Convert string IDs to appropriate type (int or UUID)
    parsed_ids = [_parse_point_id(pid) for pid in point_ids]

    results = await client.retrieve(
        collection_name=collection_name,
        ids=parsed_ids,
        with_payload=True,
        with_vectors=False,
    )

    return [
        SearchResult(
            id=str(point.id),
            score=1.0,  # Retrieved by ID, no score
            content=point.payload.get("text", point.payload.get("content", "")) if point.payload else "",
            payload=point.payload or {},
        )
        for point in results
    ]


async def get_document_chunks(
    collection_name: str,
    document_id: str,
) -> List[SearchResult]:
    """Get all chunks belonging to a specific document.

    Args:
        collection_name: Name of the collection.
        document_id: The document_id to retrieve all chunks for.

    Returns:
        List of SearchResult objects ordered by chunk_index.
    """
    client = await get_qdrant_client()

    # Scroll through all chunks with this document_id
    results = []
    offset = None

    while True:
        response, next_offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in response:
            payload = point.payload or {}
            results.append(SearchResult(
                id=str(point.id),
                score=1.0,
                content=payload.get("text", payload.get("content", "")),
                payload=payload,
            ))

        if next_offset is None:
            break
        offset = next_offset

    # Sort by chunk_index
    results.sort(key=lambda r: r.payload.get("chunk_index", 0))
    return results


async def scroll(
    collection_name: str,
    limit: int = 20,
    offset: str | None = None,
    filter_params: QdrantFilter | None = None,
) -> Tuple[List[SearchResult], Optional[str]]:
    """Scroll through points in a collection with optional filtering.

    Args:
        collection_name: Name of the collection.
        limit: Number of points to return.
        offset: Offset from previous scroll call for pagination.
        filter_params: Optional filters to apply.

    Returns:
        Tuple of (list of results, next_offset for pagination).
    """
    client = await get_qdrant_client()
    qdrant_filter = build_qdrant_filter(filter_params)

    response, next_offset = await client.scroll(
        collection_name=collection_name,
        scroll_filter=qdrant_filter,
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for point in response:
        payload = point.payload or {}
        results.append(SearchResult(
            id=str(point.id),
            score=1.0,
            content=payload.get("text", payload.get("content", "")),
            payload=payload,
        ))

    return results, str(next_offset) if next_offset else None


async def recommend(
    collection_name: str,
    positive_ids: List[str],
    negative_ids: List[str] | None = None,
    limit: int = 10,
    filter_params: QdrantFilter | None = None,
) -> List[SearchResult]:
    """Find points similar to positive examples and dissimilar to negative examples.

    Args:
        collection_name: Name of the collection.
        positive_ids: Point IDs to find similar results to.
        negative_ids: Optional point IDs to find dissimilar results from.
        limit: Maximum number of results to return.
        filter_params: Optional filters to apply.

    Returns:
        List of SearchResult objects with recommended points.
    """
    client = await get_qdrant_client()
    qdrant_filter = build_qdrant_filter(filter_params)

    # Convert string IDs to appropriate type (int or UUID)
    parsed_positive = [_parse_point_id(pid) for pid in positive_ids]
    parsed_negative = [_parse_point_id(pid) for pid in (negative_ids or [])] if negative_ids else None

    # Use query_points with RecommendQuery
    recommend_query = RecommendQuery(
        recommend=RecommendInput(
            positive=parsed_positive,
            negative=parsed_negative,
        )
    )

    results = await client.query_points(
        collection_name=collection_name,
        query=recommend_query,
        limit=limit,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [_scored_point_to_result(point) for point in results.points]


async def count(
    collection_name: str,
    filter_params: QdrantFilter | None = None,
) -> int:
    """Count points in a collection matching the filter.

    Args:
        collection_name: Name of the collection.
        filter_params: Optional filters to apply.

    Returns:
        Number of matching points.
    """
    client = await get_qdrant_client()
    qdrant_filter = build_qdrant_filter(filter_params)

    result = await client.count(
        collection_name=collection_name,
        count_filter=qdrant_filter,
        exact=True,
    )

    return result.count
