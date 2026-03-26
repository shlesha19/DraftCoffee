FROM python:3.11-slim

WORKDIR /app

COPY src/ /app

RUN pip install --no-cache-dir \
    "a2a-sdk[http-server]>=0.3.0" \
    click>=8.1.8 \
    httpx>=0.28.1 \
    openai>=1.57.0 \
    pydantic>=2.11.4 \
    python-dotenv>=1.1.0 \
    uvicorn>=0.34.2 \
    # MANTHA pipeline deps
    pandas>=2.0 \
    openpyxl>=3.1 \
    xlrd>=2.0 \
    requests>=2.31 \
    matplotlib>=3.8 \
    seaborn>=0.13 \
    reportlab>=4.1 \
    google-auth>=2.28 \
    google-auth-oauthlib>=1.2 \
    google-api-python-client>=2.120

ENV PYTHONUNBUFFERED=1

CMD ["python", "__main__.py", "--host", "0.0.0.0", "--port", "5000"]
