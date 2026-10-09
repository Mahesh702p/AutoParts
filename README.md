# AutoParts-Agent: Enterprise Inventory Optimization Engine & Synthetic Data Suite

**Project Title:** *Development of a Micro Language Model for Automotive & Chemical Manufacturing Supply Chains using PyTorch Framework: AutoParts-Agent — Multi-Agent PyTorch Framework for Just-In-Time (JIT) Components Inventory Reordering and Supplier Risk Scoring.*

---

## 1. System Overview

This directory (`AutoParts/`) contains the complete implementation and verified results for the **Inventory Optimization & Demand Forecasting Engine**:
1. **The Comprehensive 7-Table Industrial Dataset (`data/`)**: Tailored specifically for multi-tier automotive and chemical manufacturing environments, modeled with realistic demand, lead-time variance, supplier risk, and Bill of Materials (BOM) multi-echelon dependencies.
2. **The Production Evaluation Pipeline (`run_system_evaluation.py`)**: A dual-engine forecaster and policy simulator comparing the **Proposed Hybrid System** against the **Conventional Static ERP Baseline**.
3. **Empirical Results (`results/`)**: Precomputed ledgers and summary metrics over a strict 2-year out-of-sample test window (2021–2022 / 730 operational days).

---

## 2. Directory Structure

```text
AutoParts/
├── data/                                 # Complete 7-table synthetic industrial dataset
│   ├── 01_sku_master.csv                 # 82 Catalog SKUs (52 Automotive, 30 Chemical)
│   ├── 02_supplier_master.csv            # 15 Tier-1 & Tier-2 industrial suppliers
│   ├── 03_demand_history.csv             # 149,732 daily demand records (2018–2022)
│   ├── 04_inventory_snapshots.csv        # 149,732 daily stock snapshots
│   ├── 05_purchase_orders.csv            # 10,578 historical purchase orders
│   ├── 06_bom_tree.csv                   # Multi-echelon Bill of Materials relationships
│   └── 07_supplier_news_feed.csv         # 286 unstructured risk & disruption events
├── results/                              # Output artifacts from the evaluation pipeline
│   ├── simulation_kpi_summary.json       # Domain-level summary metrics
│   └── sku_performance_breakdown.csv     # Granular, per-SKU ledger (82 rows)
├── models/                               # Reserved for saved PyTorch / LightGBM weights
├── docs/                                 # Mathematical specifications & architecture docs
├── run_system_evaluation.py              # Main executable evaluation & benchmarking script
└── README.md                             # System documentation
```

---

## 3. Dataset Dictionary (`AutoParts/data/`)

| File Name | Records | Key Schema Columns | Description |
| :--- | :--- | :--- | :--- |
| **`01_sku_master.csv`** | 82 rows | `sku_id`, `category`, `unit_cost`, `holding_cost_per_day`, `order_fixed_cost`, `stockout_penalty_per_unit`, `shelf_life_days`, `lead_time_mean_days`, `lead_time_std_days` | Catalog master covering 52 automotive components and 30 industrial chemicals with specific economic parameters. |
| **`02_supplier_master.csv`** | 15 rows | `supplier_id`, `supplier_name`, `tier`, `country`, `state`, `historical_fill_rate`, `on_time_delivery_rate`, `financial_stability_score` | Industrial vendors across automotive hubs (Pune, Gurgaon, Chennai) and chemical corridors (Gujarat). |
| **`03_demand_history.csv`** | 149,732 rows | `sku_id`, `date`, `daily_demand`, `actual_consumption`, `day_of_week`, `month`, `is_weekend` | 5 years of daily continuous demand records (Jan 1, 2018 to Dec 31, 2022). |
| **`04_inventory_snapshots.csv`** | 149,732 rows | `sku_id`, `date`, `on_hand_inventory`, `pipeline_inventory`, `inventory_position`, `backlog_units` | Daily operational stock logs tracked across the 5-year timeline. |
| **`05_purchase_orders.csv`** | 10,578 rows | `po_id`, `sku_id`, `supplier_id`, `order_date`, `expected_delivery_date`, `actual_delivery_date`, `order_qty`, `delivered_qty`, `status` | Purchase order transaction logs with realistic transit delays and supplier fill rates. |
| **`06_bom_tree.csv`** | 28 rows | `assembly_id`, `component_sku_id`, `quantity_per_assembly`, `criticality_tier` | Multi-echelon parent-child dependencies across 8 primary automotive assemblies. |
| **`07_supplier_news_feed.csv`** | 286 rows | `article_id`, `date`, `supplier_id`, `headline`, `content`, `sentiment_polarity`, `risk_category` | Unstructured textual news feed containing disruption, logistics, and compliance events for supplier risk modeling. |

---

## 4. System Architecture: Proposed Hybrid System

The system avoids a one-size-fits-all approach by separating forecasting and replenishment logic based on domain physics:

```
                                  ┌───────────────────────────┐
                                  │   82 Catalog SKU Items    │
                                  └─────────────┬─────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
      ┌───────────────────────────────┐                     ┌───────────────────────────────┐
      │     Automotive (52 SKUs)      │                     │      Chemical (30 SKUs)       │
      │   High BOM Dependency & JIT   │                     │    Shelf-Life & Perishability │
      └──────────────┬────────────────┘                     └──────────────┬────────────────┘
                     │                                                     │
                     ▼                                                     ▼
      ┌───────────────────────────────┐                     ┌───────────────────────────────┐
      │  Forecaster: PyTorch LSTM     │                     │   Forecaster: LightGBM        │
      │  Sequential memory & seasonal │                     │   Tabular rolling statistics  │
      │  production calendar cycles   │                     │   & consumption momentum      │
      └──────────────┬────────────────┘                     └──────────────┬────────────────┘
                     │                                                     │
                     ▼                                                     ▼
      ┌───────────────────────────────┐                     ┌───────────────────────────────┐
      │  Policy: Continuous (s, S)    │                     │  Policy: Continuous (r, Q)    │
      │  Dynamic reorder point 's' +  │                     │  Safety stock + lot-by-lot    │
      │  target ceiling 'S' with BOM  │                     │  hazard decay tracking        │
      │  protection & batching        │                     │                               │
      └──────────────┬────────────────┘                     └──────────────┬────────────────┘
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Daily Execution Simulator │
                                  │  (2021–2022 / 730 Days)   │
                                  └───────────────────────────┘
```

### Automotive Domain (52 SKUs)
* **Forecaster:** Deep PyTorch LSTM capturing sequential temporal dependencies, weekday production spikes, and cyclical assembly demand.
* **Policy:** Continuous Review $(s, S)$ Policy with:
  * Dynamic Reorder Point $s = \hat{D}_L + z_{\alpha} \cdot \sigma_L + \text{BOM\_Buffer}$
  * Target Maximum Stock Level $S = s + \max(\text{EOQ}, \text{Batch\_Floor})$
  * *Advantage:* Continuous monitoring prevents the massive inventory collapses caused by periodic/weekly reviews, while order batching prevents excessive transaction overhead.

### Chemical Domain (30 SKUs)
* **Forecaster:** LightGBM Regressor using rolling window features (7-day, 14-day, 28-day demand statistics, trend slopes, and batch consumption indicators).
* **Policy:** Continuous $(r, Q)$ with Lot-by-Lot Hazard Decay:
  * Reorder Point $r = \hat{D}_L + z_{\alpha} \cdot \sigma_L$
  * Reorder Quantity $Q = \text{EOQ}$
  * Active monitoring of batch expiration dates to trigger reorders before stock spoilages occur.

---

## 5. Baseline: Conventional Static ERP

The baseline represents standard industry ERP setups (e.g., legacy SAP / Oracle static profiles):
* **Forecasting:** 30-day static Simple Moving Average (SMA).
* **Reorder Logic:** Static Reorder Point (Fixed ROP) calculated once using historical average demand and static nominal lead time.
* **Order Quantity:** Fixed Economic Order Quantity (EOQ).
* **Limitation:** Fails to adapt to supplier transit variance, assembly BOM synchronization, or shelf-life hazard degradation, leading to frequent line-stoppage stockouts.

---

## 6. How to Run the Pipeline

Execute the self-contained script using Python:

```bash
# Using the local project environment
/home/mahesh/Desktop/uv-packages/.venv/bin/python AutoParts/run_system_evaluation.py

# Or standard Python 3.10+ (requires: torch, lightgbm, scikit-learn, pandas, rich)
python3 AutoParts/run_system_evaluation.py
```

Execution takes approximately **15–20 seconds** on a standard CPU.

---

## 7. Benchmark Results: Proposed Engine vs. Static ERP Baseline

Evaluation conducted over the strict out-of-sample test window: **2021-01-01 to 2022-12-31 (730 operational days)** across all 82 catalog items.

### Automotive Domain (52 Components)

| Metric | Conventional Static ERP | Proposed Hybrid Engine | Empirical Improvement |
| :--- | :---: | :---: | :---: |
| **Cycle Service Level (Fill Rate)** | 95.07% | **99.66%** | **+4.59%** |
| **Total Assembly Stockout Days** | 1,623 days | **112 days** | **-93.1% (-1,511 days)** |
| **Purchase Orders Dispatched** | 1,202 POs | **917 POs** | **-23.7% (-285 POs)** |
| **Warehouse Holding Cost** | ₹7,48,814 | ₹11,32,272 | +51.2% (Calculated buffer) |
| **Procurement Transaction Fees** | ₹6,01,000 | ₹4,58,500 | -23.7% |
| **Stockout Penalties (Line Stoppage)** | ₹1,64,74,725 | ₹12,02,535 | **-92.7%** |
| **Total Domain Operating Cost** | **₹1,78,24,539** | **₹27,93,307** | **-84.3% (₹1.50 Crore saved)** |

### Chemical Domain (30 Materials)

| Metric | Conventional Static ERP | Proposed Hybrid Engine | Empirical Improvement |
| :--- | :---: | :---: | :---: |
| **Cycle Service Level (Fill Rate)** | 90.88% | **99.23%** | **+8.36%** |
| **Total Chemical Stockout Days** | 2,054 days | **169 days** | **-91.8% (-1,885 days)** |
| **Purchase Orders Dispatched** | 664 POs | **502 POs** | **-24.4% (-162 POs)** |
| **Warehouse Holding Cost** | ₹2,33,312 | ₹3,83,139 | +64.2% |
| **Expired Chemical Spoilage Loss** | ₹0 | ₹0 | ₹0 (Zero spoilage) |
| **Line Disruption Penalty** | ₹1,16,92,180 | ₹8,49,960 | **-92.7%** |
| **Total Domain Operating Cost** | **₹1,22,57,492** | **₹14,84,099** | **-87.9% (₹1.07 Crore saved)** |

### Key Economic Takeaways
1. **Stockout Reduction:** Reduced stockout days by **>91%** across both sectors, directly protecting JIT manufacturing assembly lines from expensive downtime.
2. **Total Cost Reduction:** Reduced automotive supply chain costs by **84.3%** and chemical supply chain costs by **87.9%**.
3. **Ordering Efficiency:** Dispatched ~24% fewer purchase orders through intelligent dynamic batching, lowering procurement transaction overhead.

---

## 8. Modular System Integration

This inventory optimization subsystem integrates directly into the broader AutoParts-Agent architecture:
* **Supplier Risk Scoring Engine:** Ingests `data/07_supplier_news_feed.csv` and `data/02_supplier_master.csv` into a PyTorch Micro Language Model (e.g., DistilBERT/LoRA) to dynamically adjust supplier lead-time buffers based on real-time disruption sentiment.
* **Multi-Agent Orchestration:** Feeds the dynamic reorder points and safety stock signals into an automated multi-agent framework that coordinates PO generation, supplier selection, and inventory reconciliation.
