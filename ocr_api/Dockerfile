FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY ocr_api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Run the FastAPI app
CMD ["uvicorn", "ocr_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
