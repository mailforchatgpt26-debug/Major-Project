# GNN-Based Trade Forecasting System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
[![Redis](https://img.shields.io/badge/Redis-Cache-red.svg)](https://redis.io/)

A state-of-the-art framework for predicting bilateral trade flows and analyzing supply chain risks using **Graph Neural Networks (GNNs)**. This system integrates macroeconomic indicators (World Bank) with real-time global news sentiment (GDELT) to forecast export potential and alert on supply chain disruptions.

---

## 📖 Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage Workflow](#-usage-workflow)
  - [1. Data Pipeline (ETL)](#1-data-pipeline-etl)
  - [2. Model Training](#2-model-training)
  - [3. Automated Pipelines](#3-automated-pipelines)
  - [4. API Backend](#4-api-backend)
  - [5. Dashboard](#5-dashboard)
- [Tech Stack](#-tech-stack)

---

## 🏗 System Architecture

The project follows a modular architecture where data processing, modeling, and serving are decoupled:

1.  **Data Ingestion Layer**: Fetches structured trade data (UN Comtrade, World Bank) and unstructured news signals (GDELT Project) via Google BigQuery.
2.  **Graph Construction**: Converts tabular data into temporal graph snapshots where:
    * **Nodes**: Countries (Features: GDP, Inflation, Manufacturing Output).
    * **Edges**: Trade relationships (Features: Distance, FTA, Sentiment, Lagged Exports).
3.  **Model Layer**: A **Graph Attention Network (GAT)** that learns spatial and temporal dependencies to predict future edge attributes (trade values).
4.  **Pipeline Layer**: Automated schedulers (`src/pipelines/`) that periodically fetch new articles, compute sentiment scores, and update the graph.
5.  **Serving Layer**: A FastAPI backend backed by **Redis** for high-performance caching of predictions and alerts.
6.  **Presentation Layer**: A Next.js dashboard for interactive visualization of global trade networks.

---

## 🚀 Key Features

* **Graph Attention Networks (GAT)**: Utilizes attention mechanisms to dynamically weigh the importance of trade partners.
* **Multi-Modal Data Fusion**: Combines hard economic data with soft sentiment signals from millions of news articles.
* **Real-Time Risk Alerts**: Monitors global events to trigger alerts when sentiment shocks (negative news spikes) predict trade volatility.
* **Automated Pipelines**: Self-healing cron jobs that keep data fresh without manual intervention.
* **Explainable AI (XAI)**: Decomposes predictions to show which factors (e.g., "GDP Growth" vs. "Negative News") drove the forecast.
* **Interactive Dashboard**: A modern UI offering geospatial visualizations, prediction tables, and drill-down analysis per country.

---

## 📂 Repository Structure

The codebase strictly separates core library logic (`src/`) from operational scripts (`scripts/`).

```text
├── configs/                 # YAML Control Center
│   ├── model_config.yaml    # GAT hyperparameters (layers, heads, dropout)
│   ├── pipeline_config.yaml # Data sources, alert thresholds, & API keys
│   └── features.yaml        # Feature engineering definitions
├── dashboard/               # Next.js Frontend Application
│   └── src/                 # React components, pages, and hooks
├── data/                    # Data Lake (Raw, Processed, Scalers)
├── models/                  # Saved model checkpoints (*.pt)
├── scripts/                 # Operational Entry Points
│   ├── preprocess_data.py   # ETL: Raw Data -> Graph Snapshots
│   ├── train_model.py       # Training Loop
│   ├── scheduler_service.py # Cron: Runs periodic updates
│   └── quickstart.py        # Health Check
├── src/                     # Core Library
│   ├── api/                 # FastAPI routes & Redis caching
│   ├── data/                # Graph builders & loaders
│   ├── models/              # PyTorch GNN architecture (gnn.py)
│   ├── pipelines/           # Automation Logic
│   │   ├── gdelt_fetcher.py # BigQuery Interface
│   │   ├── sentiment_analyzer.py # Tone/Sentiment Engine
│   │   └── gdelt_article_scheduler.py # Job Orchestrator
│   └── utils/               # Database, logging, helpers
└── requirements.txt         # Python dependencies
```

---

## 📋 Prerequisites

* **Python**: 3.10+
* **Node.js**: 18+ (for Dashboard)
* **PostgreSQL**: Primary storage for structured trade data.
* **Redis**: **Required** for caching API responses and real-time alerts.
* **Google Cloud Platform**: Service Account with **BigQuery Data Viewer** role (for GDELT news ingestion).

---

## ⚙️ Installation

### 1. Backend Setup

```bash
# Clone the repository
git clone [https://github.com/your-username/gnn-trade-forecasting.git](https://github.com/your-username/gnn-trade-forecasting.git)
cd gnn-trade-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```
### 2. Frontend Setup

```bash
cd dashboard/src
npm install
# or
pnpm install
```
### 3. Service Setup (Docker Recommended)

```bash
# Start Redis and Postgres
docker run --name trade-redis -p 6379:6379 -d redis
docker run --name trade-postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres
```
---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```ini
# Database & Cache
DATABASE_URL=postgresql://postgres:password@localhost:5432/trade_db
REDIS_URL=redis://localhost:6379/0

# Google Cloud (Critical for News Data)
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```
### YAML Configuration

- `configs/pipeline_config.yaml`: Defines which commodities to track (e.g., "Pharmaceuticals", "Textiles") and GDELT keywords.
- `configs/model_config.yaml`: Adjusts the GNN depth and training epochs.
  
---

## 🏃 Usage Workflow

### 1. Data Pipeline (ETL)

First, ingest raw data and build the graph snapshots.

```bash
# Validate connections
python scripts/quickstart.py

# Run the ETL pipeline
python scripts/preprocess_data.py
```
### 2. Model Training

Train the Graph Neural Network. Artifacts are saved to `models/`.

```bash
python scripts/train_model.py
```
### 3. Automated Pipelines

To enable real-time news monitoring, start the scheduler. This runs the scripts found in `src/pipelines/` to fetch GDELT data every 15 minutes.

```bash
python scripts/scheduler_service.py
```
### 4. API Backend

Start the FastAPI server. This serves the trained model and cached alerts.

```bash
python src/api/main.py
```
### 5. Dashboard

Launch the visualization interface.

```bash
cd dashboard/src
npm run dev
```

## 💻 Tech Stack

| Domain | Technologies |
|--------|-------------|
| Machine Learning | PyTorch, PyTorch Geometric, Scikit-Learn |
| Backend API | FastAPI, Uvicorn |
| Caching / Msg Queue | Redis (Critical for low-latency alerts) |
| Data Processing | Pandas, NumPy, Google BigQuery (GDELT) |
| Automation | APScheduler (src/pipelines/) |
| Frontend | Next.js 14, React, Tailwind CSS v4, Recharts |
| Infrastructure | Docker, Git |

# Major_project
