#!/bin/bash

# GNN Trade Forecasting System - COMPREHENSIVE RUN SCRIPT
# This script manages Preprocessing, Training, Backend, and Frontend.

PROJECT_ROOT=$(pwd)
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"
export PYTHONPATH=$PROJECT_ROOT

echo "=========================================================="
echo "🚀 INITIALIZING GNN TRADE FORECASTING ECOSYSTEM"
echo "=========================================================="

# 1. Environment & Folder Check
echo "🔍 Checking directories..."
mkdir -p logs models data/processed

# 2. Python Dependencies Check
if [ ! -d "venv" ]; then
    echo "❌ venv not found! Creating virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
else
    echo "✅ Python Venv: OK"
fi

# 3. Force Cleanup (8000/3000)
echo "🧹 Force killing existing processes on ports 8000 & 3000..."
lsof -ti:8000,3000 | xargs kill -9 2>/dev/null || true
sleep 1

# 4. Data Preprocessing
echo -e "\n📊 Step 1/4: Data Preprocessing..."
$VENV_PYTHON scripts/preprocess_data.py
if [ $? -ne 0 ]; then
    echo "❌ Preprocessing failed. Check logs/preprocessing.log"
    exit 1
fi

# 5. Model Initialization (Standard + Causal)
echo -e "\n🧠 Step 2/4: Checking GNN Models..."

if [ ! -f "models/gnn_working.pt" ]; then
    echo "🏗️  Baseline GNN Model missing. Training now..."
    $VENV_PYTHON scripts/train_model.py
else
    echo "✅ Baseline GNN: READY"
fi

# Initialize the new Causal CausalEngine logic snapshots
echo "🧬 Initializing Causal Reasoning Engine snapshots..."
$VENV_PYTHON scripts/train_causal.py

# 6. Start FastAPI Backend
echo -e "\n📡 Step 3/4: Launching AI Backend..."
nohup $VENV_PYTHON src/api/main.py > logs/api.log 2>&1 &
API_PID=$!

echo "⏳ Waiting for AI to initialize (Neural Graph Snapshots)..."
sleep 5

if ps -p $API_PID > /dev/null; then
    echo "✅ AI Backend: LIVE (PID: $API_PID)"
else
    echo "❌ AI Backend failed. See logs/api.log"
    tail -n 15 logs/api.log
    exit 1
fi

# 7. Start Next.js Dashboard
echo -e "\n🖥️  Step 4/4: Launching Dashboard..."
cd dashboard/src

if [ ! -d "node_modules" ]; then
    echo "📦 node_modules missing. Installing dashboard dependencies..."
    npm install
fi

echo -e "\n✨ ALL SYSTEMS GO! ✨"
echo "----------------------------------------------------------"
echo "🌐 DASHBOARD: http://localhost:3000"
echo "📡 AI BACKEND: http://localhost:8000"
echo "🧬 ENGINE: Counterfactual Causal Simulation ENABLED"
echo "----------------------------------------------------------"

npm run dev
