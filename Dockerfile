# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Expose the port your app runs on (8888)
EXPOSE 8888

CMD ["python", "app.py"]
