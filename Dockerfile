FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY models/ /app/models/
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:app"]
