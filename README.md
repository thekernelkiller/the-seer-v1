```bash
make run
```

The uvicorn server will start on `http://localhost:8080` and streamlit server on `http://localhost:8051`

## 📡 API Usage

### Quick Analysis
```bash
# Start analysis
curl -X POST "http://localhost:8080/api/v1/analysis/start" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "RELIANCE:NS",
    "analysis_type": "comprehensive",
    "time_horizon": "medium_term"
  }'

# Check status (use session_id from above)
curl "http://localhost:8080/api/v1/analysis/status/{session_id}"

# Get results
curl "http://localhost:8080/api/v1/analysis/result/{session_id}"
```

### Synchronous Analysis
```bash
# Wait for completion (up to 5 minutes)
curl -X POST "http://localhost:8080/api/v1/analysis/analyze?wait_for_completion=true" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "TCS:NS",
    "analysis_type": "comprehensive"
  }'
```

### Batch Analysis
```bash
curl -X POST "http://localhost:8080/api/v1/analysis/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["RELIANCE:NS", "TCS:NS", "INFY:NS"],
    "analysis_type": "quick",
    "time_horizon": "medium_term"
  }'
```
