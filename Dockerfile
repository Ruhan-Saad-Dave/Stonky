FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

# Install uv
RUN pip install uv

# Install dependencies using uv
RUN uv pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app"]