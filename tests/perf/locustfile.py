from __future__ import annotations

import random
from locust import HttpUser, between, task


PARTNERS = ["USA", "DEU", "FRA", "JPN", "BRA", "ZAF", "VNM"]
SECTOR = "pharma"
MONTH = "2025-01"


class TradeApiUser(HttpUser):
    """
    Locust load profile for the Trade Flow backend.

    Run example:
      locust -f tests/perf/locustfile.py --host http://127.0.0.1:8000
    """

    wait_time = between(1, 4)

    @task(2)
    def health(self):
        self.client.get("/health", name="/health")

    @task(5)
    def predictions(self):
        self.client.get(
            f"/api/predictions?sector={SECTOR}&month={MONTH}",
            name="/api/predictions",
        )

    @task(3)
    def news(self):
        self.client.get(
            f"/api/news?sector={SECTOR}&month={MONTH}",
            name="/api/news",
        )

    @task(2)
    def explainability(self):
        partner = random.choice(PARTNERS)
        self.client.get(
            f"/api/explainability?partner={partner}",
            name="/api/explainability",
        )

    @task(1)
    def simulate(self):
        partner = random.choice(PARTNERS)
        payload = {
            "target_country": partner,
            "feature": random.choice(["sentiment", "lag1", "distance"]),
            "change_percent": random.choice([-10.0, -5.0, 5.0, 10.0]),
            "sector": SECTOR,
            "month": MONTH,
        }
        self.client.post("/api/v1/simulate", json=payload, name="/api/v1/simulate")

