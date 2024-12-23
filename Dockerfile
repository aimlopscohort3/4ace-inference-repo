# Use a slim Python image
FROM python:3.8-slim-buster

# Set environment variables to improve performance and ensure consistency
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TEMPLATE_DIR=/app/templates

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for the app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose application and Prometheus metrics ports
EXPOSE 8080 8000

# Add a non-root user for running the application
RUN useradd -m appuser
USER appuser

# Set default entrypoint and command
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8080"]