FROM python:3.9-slim

WORKDIR /app

COPY . /app
COPY app/training.html /app/static/training.html

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /app/models /app/data/csv

# Copy the CSV file into the container
COPY app/data/csv/koi_data_100_samples.csv /app/data/csv/

# Thêm dòng này để tạo volume
VOLUME /app/data

EXPOSE 9001
# Run koi_health_predictor.py before starting the FastAPI server
CMD ["sh", "-c", "python /app/app/koi_health_predictor.py && uvicorn app.main:app --host 0.0.0.0 --port 9001"]
