"""
Model Diagnostic - Verify predictions are reasonable
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.loaders import GraphDataLoader, TemporalDataset
from src.models.gnn import TradeGNN

print("="*60)
print("🔬 MODEL DIAGNOSTIC")
print("="*60)

# Load model
model_path = list(Path("models").glob("gnn_working.pt"))[-1]
print(f"\n📦 Loading model: {model_path.name}")

checkpoint = torch.load(model_path, map_location='cpu')
config = checkpoint['config']

model = TradeGNN(
    num_node_features=config['num_node_features'],
    num_edge_features=config['num_edge_features'],
    hidden_dim=128,
    num_layers=3,
    dropout=0.3,
    heads=4
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print("✓ Model loaded successfully")

# Load test data
print("\n📊 Loading test data...")
loader = GraphDataLoader("data/processed")
graphs = loader.create_temporal_graphs()
dataset = TemporalDataset(graphs)
_, _, test_graphs = dataset.split(train=0.7, val=0.15)

print(f"✓ {len(test_graphs)} test graphs")

# Make predictions
print("\n🎯 MAKING PREDICTIONS")
print("="*60)

all_preds = []
all_labels = []

with torch.no_grad():
    for i, graph in enumerate(test_graphs):
        out = model(graph.x, graph.edge_index, graph.edge_attr)
        
        preds = out.numpy()
        labels = graph.y.numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Sample predictions
print(f"\nTotal predictions: {len(all_preds)}")
print(f"\nSample Predictions (first 20):")
print("-"*60)
print(f"{'Actual (log)':<15} {'Predicted (log)':<18} {'Actual ($)':<15} {'Predicted ($)':<15} {'Error %'}")
print("-"*60)

for i in range(min(20, len(all_preds))):
    actual_log = all_labels[i]
    pred_log = all_preds[i]
    
    actual_usd = np.expm1(actual_log)
    pred_usd = np.expm1(pred_log)
    
    error_pct = abs(actual_usd - pred_usd) / actual_usd * 100
    
    print(f"{actual_log:14.2f}  {pred_log:17.2f}  ${actual_usd:13,.0f}  ${pred_usd:13,.0f}  {error_pct:6.1f}%")

# Statistics
print("\n" + "="*60)
print("📈 PREDICTION STATISTICS")
print("="*60)

# Log scale
print("\nOn Log Scale:")
print(f"  Actual range:    {all_labels.min():.2f} to {all_labels.max():.2f}")
print(f"  Predicted range: {all_preds.min():.2f} to {all_preds.max():.2f}")
print(f"  Mean actual:     {all_labels.mean():.2f}")
print(f"  Mean predicted:  {all_preds.mean():.2f}")

# Original scale
actual_orig = np.expm1(all_labels)
pred_orig = np.expm1(all_preds)

print("\nOn Original Scale ($):")
print(f"  Actual range:    ${actual_orig.min():,.0f} to ${actual_orig.max():,.0f}")
print(f"  Predicted range: ${pred_orig.min():,.0f} to ${pred_orig.max():,.0f}")
print(f"  Median actual:   ${np.median(actual_orig):,.0f}")
print(f"  Median predicted: ${np.median(pred_orig):,.0f}")

# Error analysis
errors = np.abs(all_labels - all_preds)
print("\nError Distribution:")
print(f"  Mean error (log): {errors.mean():.2f}")
print(f"  Median error (log): {np.median(errors):.2f}")
print(f"  90th percentile: {np.percentile(errors, 90):.2f}")
print(f"  Max error: {errors.max():.2f}")

# How many predictions within X% on original scale
pct_errors = np.abs((actual_orig - pred_orig) / actual_orig) * 100
print("\nAccuracy Breakdown:")
print(f"  Within 50%:  {(pct_errors <= 50).sum() / len(pct_errors) * 100:.1f}%")
print(f"  Within 100%: {(pct_errors <= 100).sum() / len(pct_errors) * 100:.1f}%")
print(f"  Within 200%: {(pct_errors <= 200).sum() / len(pct_errors) * 100:.1f}%")
print(f"  Within 500%: {(pct_errors <= 500).sum() / len(pct_errors) * 100:.1f}%")

# R² score
from sklearn.metrics import r2_score
r2 = r2_score(all_labels, all_preds)

print("\n" + "="*60)
print("✅ MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"R² Score: {r2:.4f}")
print(f"MAE (log scale): {np.mean(np.abs(all_labels - all_preds)):.2f}")
print(f"Median % Error: {np.median(pct_errors):.1f}%")
print(f"\n{'✅ MODEL IS WORKING WELL!' if r2 > 0.7 else '⚠️  Model needs improvement'}")
print("="*60)