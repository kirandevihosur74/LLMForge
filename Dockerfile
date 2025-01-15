# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY ./api /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn transformers tensorflow

# Expose the API port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]