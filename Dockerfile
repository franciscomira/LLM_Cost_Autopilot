FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./

# Install Python dependencies (editable install picks up src/autopilot package)
RUN pip install --no-cache-dir -e "." 2>/dev/null || \
    pip install --no-cache-dir \
        httpx pyyaml pydantic psutil anthropic \
        fastapi "uvicorn[standard]" streamlit python-dotenv rich

# Copy source
COPY . .

# SQLite data volume mount point
RUN mkdir -p data

# Drop root — create a non-privileged user and hand over the app directory
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser \
    && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "autopilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
