# """
# Data Diagnostic and Fix Script
# Run this first to understand your data structure
# """
# import pandas as pd
# import json
# from pathlib import Path

# print("="*60)
# print("🔍 DATA DIAGNOSTIC")
# print("="*60)

# # 1. Check nodes.csv
# print("\n1️⃣  NODES.CSV ANALYSIS")
# print("-"*60)
# nodes_df = pd.read_csv("data/processed/nodes.csv")
# print(f"Total rows: {len(nodes_df):,}")
# print(f"Columns: {list(nodes_df.columns)}")
# print(f"\nFirst 5 rows:")
# print(nodes_df.head())
# print(f"\nUnique countries: {nodes_df['country'].nunique() if 'country' in nodes_df.columns else 'N/A'}")
# print(f"Node ID range: {nodes_df['node_id'].min()} to {nodes_df['node_id'].max()}")

# # 2. Check edges.csv
# print("\n2️⃣  EDGES.CSV ANALYSIS")
# print("-"*60)
# edges_df = pd.read_csv("data/processed/edges.csv")
# print(f"Total rows: {len(edges_df):,}")
# print(f"Columns: {list(edges_df.columns)}")
# print(f"\nFirst 5 rows:")
# print(edges_df.head())

# # Check node IDs in edges
# print(f"\nSource node ID range: {edges_df['source_node_id'].min()} to {edges_df['source_node_id'].max()}")
# print(f"Target node ID range: {edges_df['target_node_id'].min()} to {edges_df['target_node_id'].max()}")

# # Check for sentiment data
# if 'sentiment_score' in edges_df.columns:
#     print(f"\n✓ Sentiment scores present")
#     print(f"  Range: {edges_df['sentiment_score'].min():.3f} to {edges_df['sentiment_score'].max():.3f}")
#     print(f"  Mean: {edges_df['sentiment_score'].mean():.3f}")
#     print(f"  Non-zero: {(edges_df['sentiment_score'] != 0).sum():,} ({(edges_df['sentiment_score'] != 0).sum()/len(edges_df)*100:.1f}%)")
# else:
#     print("\n❌ No sentiment_score column!")

# # Check for trade values
# if 'trade_value_usd' in edges_df.columns:
#     print(f"\n✓ Trade values present")
#     print(f"  Range: ${edges_df['trade_value_usd'].min():,.0f} to ${edges_df['trade_value_usd'].max():,.0f}")
#     print(f"  Mean: ${edges_df['trade_value_usd'].mean():,.0f}")
# else:
#     print("\n❌ No trade_value_usd column!")

# # 3. Check node_mapping.json
# print("\n3️⃣  NODE_MAPPING.JSON ANALYSIS")
# print("-"*60)
# mapping_path = Path("data/processed/node_mapping.json")
# if mapping_path.exists():
#     with open(mapping_path, 'r') as f:
#         node_mapping = json.load(f)
#     print(f"Total mappings: {len(node_mapping)}")
#     print(f"First 10 mappings: {dict(list(node_mapping.items())[:10])}")
# else:
#     print("❌ node_mapping.json not found!")

# # 4. Check for mismatches
# print("\n4️⃣  MISMATCH DETECTION")
# print("-"*60)
# print(f"Nodes in nodes.csv: {len(nodes_df)}")
# print(f"Unique source nodes in edges: {edges_df['source_node_id'].nunique()}")
# print(f"Unique target nodes in edges: {edges_df['target_node_id'].nunique()}")
# print(f"Countries in node_mapping: {len(node_mapping)}")

# # Check if node IDs in edges exceed nodes.csv
# max_edge_node = max(edges_df['source_node_id'].max(), edges_df['target_node_id'].max())
# if max_edge_node >= len(nodes_df):
#     print(f"\n⚠️  WARNING: Edge node IDs ({max_edge_node}) exceed nodes.csv length ({len(nodes_df)})")
#     print("   This will cause index out of bounds errors!")

# # 5. Time period analysis
# print("\n5️⃣  TIME PERIOD ANALYSIS")
# print("-"*60)
# if 'year' in edges_df.columns and 'month' in edges_df.columns:
#     edges_df['time_key'] = edges_df['year'].astype(str) + '-' + edges_df['month'].astype(str).str.zfill(2)
#     time_periods = sorted(edges_df['time_key'].unique())
#     print(f"Time periods: {len(time_periods)}")
#     print(f"Range: {time_periods[0]} to {time_periods[-1]}")
    
#     # Edges per time period
#     period_counts = edges_df.groupby('time_key').size().describe()
#     print(f"\nEdges per period:")
#     print(f"  Min: {period_counts['min']:.0f}")
#     print(f"  Max: {period_counts['max']:.0f}")
#     print(f"  Mean: {period_counts['mean']:.0f}")
# else:
#     print("❌ No year/month columns!")

# print("\n" + "="*60)
# print("✅ DIAGNOSTIC COMPLETE")
# print("="*60)

# # 6. Recommendations
# print("\n💡 RECOMMENDATIONS:")
# print("-"*60)

# if len(node_mapping) < 10:
#     print("❌ CRITICAL: node_mapping.json has too few countries")
#     print("   → Need to rebuild node_mapping from nodes.csv")
#     print("   → Run the fix script below")

# if max_edge_node >= len(nodes_df):
#     print("❌ CRITICAL: Node ID mismatch")
#     print("   → Edge node IDs don't match nodes.csv")
#     print("   → Data preprocessing needs to be fixed")

# if 'sentiment_score' in edges_df.columns:
#     if (edges_df['sentiment_score'] == 0).sum() / len(edges_df) > 0.9:
#         print("⚠️  WARNING: >90% of sentiment scores are zero")
#         print("   → Model won't learn much from sentiment")
# else:
#     print("❌ CRITICAL: No sentiment scores in edges")
#     print("   → Need to add sentiment data")

# print("\n" + "="*60)






import pandas as pd

# Load edges
edges = pd.read_csv('data/processed/edges.csv')

# Check India exports
india_exports = edges[edges['source_iso3'] == 'IND']

print(f"Total India export records: {len(india_exports)}")
print(f"Unique partners: {india_exports['target_iso3'].nunique()}")
print(f"Sectors: {india_exports['sector'].unique()}")
print(f"\nYears available: {sorted(india_exports['year'].unique())}")

# Check a specific country pair
usa_trade = india_exports[india_exports['target_iso3'] == 'USA']
print(f"\nIndia → USA records: {len(usa_trade)}")
print(usa_trade[['year', 'month', 'sector', 'trade_value_usd', 'sentiment_norm']].tail(10))

# Check for pharmaceutical trades
pharma = india_exports[india_exports['sector'] == 'Pharmaceuticals']
print(f"\nPharmaceutical export records: {len(pharma)}")
print(f"Partners: {pharma['target_iso3'].nunique()}")

# Sample some data
print("\nSample pharmaceutical exports:")
print(pharma[['target_iso3', 'year', 'month', 'trade_value_usd']].head(20))