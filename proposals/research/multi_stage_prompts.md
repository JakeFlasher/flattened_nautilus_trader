```stage 1
**Role & Persona:**
Act as a **Principal Quantitative Scientist (PhD)** at a top-tier proprietary trading firm. Your objective is not merely to draft a strategy, but to author a comprehensive **Technical Whitepaper and System Architecture Specification** for a high-frequency volatility arbitrage system.

**Reasoning Effort:** xHigh
**Target Output Length:** Maximum Exhaustiveness (Targeting doctoral thesis depth).

---

### **1. Research Mandate: The Theoretical Framework**
**<web_search_rules>**
Perform exhaustive "Deep Dive" research. Do not settle for surface-level definitions.
1.  **Search Targets:** "Microstructure noise in crypto futures," "High-frequency volatility estimators (Garman-Klass vs. Rogers-Satchell vs. Yang-Zhang) efficiency in BTCUSDT," "Order Flow Toxicity (VPIN) using Aggregated Trades," "Funding Rate arbitrage decay models."
2.  **Synthesis:** You must synthesize these papers into a coherent mathematical model. When defining an alpha factor, you must derive it from first principles (e.g., starting from the diffusion process of the price asset) [2].
**</web_search_rules>**

---

### **2. Dataset Specification & Data Engineering**
You are constrained to the following **Local File System** architecture. You must define the precise mathematical transformation from *Raw CSV* to *Feature Vector* for every file type [1][3].

**<data_inventory>**
*   **Microstructure Features (Flow):** `data/raw/futures/daily/aggTrades/BTCUSDT-aggTrades-*.csv`
    *   *Columns:* `TradeId, Price, Quantity, First TradeId, Last TradeId, Timestamp, IsBuyerMaker`
    *   *Requirement:* Define an estimator for **Order Flow Imbalance (OFI)** and **VPIN (Volume-Synchronized Probability of Informed Trading)** using `Quantity` and `IsBuyerMaker` [3].
*   **Volatility Features (Kinematic):** `data/raw/futures/daily/klines_1m/BTCUSDT-1m-*.csv`
    *   *Requirement:* Construct a **Rogers-Satchell** estimator using $H_t$ (High), $L_t$ (Low), $O_t$ (Open), $C_t$ (Close) to capture variance independent of drift.
*   **Liquidity Features (State):** `data/raw/futures/daily/bookTicker/BTCUSDT-bookTicker-*.csv`
    *   *Requirement:* Define **Bid-Ask Spread** $\mathcal{S}_t$ and **Order Book Imbalance** $\rho_t$ derived from `best_bid_qty` vs `best_ask_qty`.
*   **Regime Features (Macro):** `fundingRate`, `metrics` (`long_short_ratio`) [3].
**</data_inventory>**

---

### **3. The Strategy Proposal (Whitepaper Requirements)**
Produce a hierarchical Markdown document. To meet the density requirements, every section must include **Formal Definitions**, **Lemmas**, and **Proof logic** where applicable.

#### **Chapter 1: Mathematical Definition of Alpha**
*   Define the *Fair Value* price process $P_t$ as a stochastic differential equation (SDE).
*   **Factor 1 (Realized Volatility):** Formulate the **Yang-Zhang estimator** (combining overnight volatility and open-close volatility). Explain why this is superior for crypto markets (24/7 trading) compared to standard close-to-close estimators [2].
*   **Factor 2 (Flow Toxicity):** Formulate the derivation of **VPIN** strictly using the `aggTrades` schema (`IsBuyerMaker` flag). Define the "Bulk Volume" bucket classification logic.
*   **Signal Combination:** Define a multivariate Kalman Filter or a Z-score weighted ensemble to combine these factors into a single target position signal $\theta_t$.

#### **Chapter 2: The NautilusTrader Architecture**
*   **Data Loading:** Provide the specific Python/Nautilus code logic for a `CSVDataLoader` that reads the specific directory structure `data/raw/futures/*` [1][4].
*   **Event-Driven Logic:** Detail the `on_quote_tick` vs. `on_bar` logic.
    *   *Logic:* How to update the Volatility Factor on `on_bar` (1m Klines) while updating the Liquidity Factor on `on_quote_tick` (BookTicker) asynchronously.
*   **Execution Algorithm:** Define a **TWAP (Time-Weighted Average Price)** algorithm with alpha-decay control. If the alpha signal strength $|\theta_t|$ exceeds a threshold, accelerate execution; otherwise, dampen it to save transaction costs.

#### **Chapter 3: Risk Management & Portfolio Construction**
*   **Covariance Modeling:** Define how to calculate the rolling covariance matrix $\Sigma_t$ of the returns if extending to multiple assets.
*   **Kelly Criterion:** Adapt the Continuous Kelly Criterion for position sizing based on the estimated volatility $\sigma_t^2$ derived in Chapter 1.
*   **Stop-Loss Logic:** Define a "Volatility-Adjusted Trailing Stop" rather than a fixed percentage.

#### **Chapter 4: Implementation Appendix (Pseudo-Code & Diagrams)**
*   **ASCII System Diagram:** Visualize the flow: `Local CSV` -> `Custom Wrangler` -> `Parquet Catalog` -> `BarAggregate` -> `Strategy Actor` -> `Risk Engine`.
*   **Data Structures:** Define the precise `CustomData` classes needed to pass signals between the Strategy and the Execution Algorithm [4].

---

### **4. Constraint Checklist**
1.  **Length & Depth:** Do not summarize. Elaborate on every equation. If an equation has assumptions (e.g., "market efficiency"), list them explicitly.
2.  **Formatting:** Use strict LaTeX for math: $$ \sigma_{RS}^2 = \frac{1}{n} \sum ... $$
3.  **Citations:** Cite papers found during search. If referencing specific NautilusTrader components (e.g., `OrderEmulator`, `Portfolio`), cite the "Nautilus Documentation" provided in the context [4].
4.  **No APIs:** Strictly use the file paths provided in the context (e.g., `data/raw/futures/daily/aggTrades/BTCUSDT-aggTrades-2023-05-16.csv`) [1].

**Output Generation:** Begin by outlining the Table of Contents for a 50-page Technical Specification, then proceed to execute Chapter 1.
```
