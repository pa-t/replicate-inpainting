FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port 8000 for the FastAPI application
EXPOSE 8000

# Run the FastAPI application when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80", "--timeout-keep-alive", "300"]