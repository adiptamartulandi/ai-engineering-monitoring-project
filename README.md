# Tweet Sentiment Analysis API

This project implements a sentiment analysis model for tweets using Logistic Regression, wrapped in a FastAPI application with monitoring capabilities using Prometheus and Grafana.

## Project Structure

```
.
├── src/
│   ├── app.py          
│   ├── train.py        
│   ├── models/         
│   │   ├── lgbm_model.py
│   │   └── __init__.py
│   ├── data/      
│   │   ├── dataset.py
│   │   └── __init__.py          
│   └── test/      
│       └── locustfile.py     
├── data/                
├── requirements.txt     
├── Dockerfile          
└── docker-compose.yml 
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python src/train.py
```

4. Start the application:

   a. For local development:
   ```bash
   uvicorn src.app:app --reload
   ```

   b. With monitoring (Prometheus + Grafana):
   ```bash
   docker-compose up --build
   ```

## API Usage

The API exposes a single endpoint for sentiment prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this new feature!"}'
```

Response format:
```json
{
    "sentiment": "positive",
    "confidence": 0.95
}
```

## Monitoring

- FastAPI Application: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default credentials: admin/admin)

### Available Metrics

- `prediction_time_seconds`: Histogram of prediction processing time
- `prediction_requests_total`: Counter of total predictions by sentiment

## Query in Prometheus

- Go to the 'Graph' tab
- Try these basic queries:
- Rate of predictions:`rate(prediction_requests_total[5m])`
- Average response time:`rate(prediction_time_seconds_sum[5m]) / rate(prediction_time_seconds_count[5m])`
- HTTP success rate:`sum(rate(http_requests_total{status="200"}[5m]))`

## Visualization in Grafana

### Connection to Prometheus

- Go to connection, Click "Add data source" and select "Prometheus".
- In the HTTP URL field, enter " http://prometheus:9090 " (this is the internal Docker network URL).
- Click "Save & Test" to verify the connection

### Creating Dashboard

- Click '+ Create' > 'Dashboard'
- Add a new panel and select Prometheus as the data source
- For basic metrics, you can use these queries:
- `prediction_requests_total` - Shows total predictions by sentiment
- `prediction_time_seconds_bucket` - Shows prediction latency distribution
- `http_requests_total` - Shows total HTTP requests by endpoint

## Performance Testing

To run performance tests:

```bash
locust -f src/test/locustfile.py
```

Access the Locust web interface at http://localhost:8089 to configure and start the test.
