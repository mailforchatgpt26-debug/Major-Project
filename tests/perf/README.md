# Locust Performance Testing

This folder contains a Locust workload for the backend API.

## Install

```bash
pip install locust
```

## Run (local backend)

```bash
locust -f tests/perf/locustfile.py --host http://127.0.0.1:8000
```

Then open:

`http://127.0.0.1:8089`

## Suggested demo profile

- Users: `20`
- Spawn rate: `5/sec`
- Duration: `2-3 min`

## Endpoints covered

- `GET /health`
- `GET /api/predictions`
- `GET /api/news`
- `GET /api/explainability`
- `POST /api/v1/simulate`

