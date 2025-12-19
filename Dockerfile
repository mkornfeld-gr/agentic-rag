FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install dependencies with uv
RUN uv sync

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the MCP server
CMD ["uv", "run", "python", "-m", "agentic_rag.mcp_server"]
