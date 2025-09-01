FROM python:3.8-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install AWS CLI using pip (no apt needed)
RUN pip install awscli --upgrade

# Expose port (if you are running Streamlit/Flask)
EXPOSE 8080

# Default command
CMD ["python3", "app.py"]
