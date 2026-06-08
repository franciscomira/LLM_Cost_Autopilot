FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./

# Install Python dependencies (no editable install — copy source below)
RUN pip install --no-cache-dir ".[standard]" 2>/dev/null || \
    pip install --no-cache-dir \
        httpx pyyaml pydantic psutil anthropic \
        fastapi "uvicorn[standard]" streamlit python-dotenv rich

# Copy source
COPY . .

# SQLite data volume mount point
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
