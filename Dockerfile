# Use a slim Python image
FROM python:3.8-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Add a non-root user
RUN useradd -m appuser
USER appuser

# Expose the application port
EXPOSE 8080

# # Set the entrypoint for flexibility
# ENTRYPOINT ["uvicorn"]

# Set the default command
CMD ["app.main:app", "--port", "8080"]