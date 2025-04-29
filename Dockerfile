FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code and assets
COPY agent.py .
COPY agent_base.py .
COPY agent_container.py .
COPY interest_model.py .
COPY config.yaml .

# Copy WebSocket server implementation files
COPY websocket_server.py .
COPY event_handlers.py .

# Create an empty examples file that will be overwritten if one exists
RUN touch /app/examples.jsonl

# Try to copy examples file (will override the empty one)
COPY examples.jsonl /app/

# Create directory for models and copy any existing fine-tuned model files
RUN mkdir -p ./fine_tuned_model/

# Create necessary directories for the fine-tuned model
RUN mkdir -p ./fine_tuned_model/1_Pooling/

# IMPORTANT: Docker doesn't support shell-style error handling with || in COPY commands
# Just use standard COPY commands and accept that some might produce warnings

# Copy the entire model directory if it exists
COPY fine_tuned_model/ ./fine_tuned_model/

# Optional: Also try to copy specific files individually
# These may produce warnings but won't fail the build if files don't exist
# because we're copying the directory above and these are just for redundancy
# COPY fine_tuned_model/classification_head.pt ./fine_tuned_model/
# COPY fine_tuned_model/model.safetensors ./fine_tuned_model/

# Make sure we have a proper README for debugging
RUN echo "This directory should contain classification_head.pt and other model files" > ./fine_tuned_model/README.md

# Copy pooling directory files if the directory exists
# This might produce a warning if the directory doesn't exist, but won't fail the build
COPY fine_tuned_model/1_Pooling/ ./fine_tuned_model/1_Pooling/

# Environment variables
ENV AGENT_ID="${AGENT_ID}"
ENV AGENT_NAME="${AGENT_NAME}"

# Set Core API URL with multiple fallback options
# The container will try these URLs in sequence
ENV CORE_API_URL="http://localhost:8888"
ENV ALTERNATE_CORE_API_URLS="http://host.docker.internal:8888,http://127.0.0.1:8888"

# Set the polling interval to check for messages
ENV POLLING_INTERVAL="5"

# WebSocket server port
ENV PORT="8000"

# Expose the WebSocket port
EXPOSE 8000

# Run the agent
CMD ["python", "agent_container.py"]
