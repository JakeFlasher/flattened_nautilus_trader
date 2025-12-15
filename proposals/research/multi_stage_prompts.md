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


```stage 2
**Role & Objective:**

You are the **Lead Algorithmic Architect** implementing the mathematical alpha defined in Chapter 1. Your goal is to author **Chapter 2: The NautilusTrader Architecture**, detailing exactly how to translate the theoretical VPIN and Volatility estimators into event-driven code using the **NautilusTrader** framework.

**Input Context (From Chapter 1):**

Assume the existence of the following defined alpha components:

1.  **$\sigma_{YZ}^2$ (Yang-Zhang Volatility):** Calculated on 1-minute bars.

2.  **$VPIN$ (Flow Toxicity):** Calculated from aggregated trades (tick-level).

3.  **$\mathcal{L}$ (Liquidity State):** Derived from BookTicker updates.

**Dataset Constraints (Strict):**

You must design the data ingestion pipeline for the following specific local file paths [1][3]:

*   **Trades:** data/raw/futures/daily/aggTrades/BTCUSDT-aggTrades-*.csv

*   **Quotes:** data/raw/futures/daily/bookTicker/BTCUSDT-bookTicker-*.csv

*   **Bars:** data/raw/futures/daily/klines_1m/BTCUSDT-1m-*.csv

---

### **Chapter 2 Requirements**

Please generate a highly technical, code-heavy (Python/Cython style) architecture section covering the following:

#### **2.1 Custom Data Loading & Wrangling**

Since we are using raw Binance CSVs locally (no API keys), you must define the **Wrangling Logic** to convert these CSVs into Nautilus internal objects [4].

*   **Schema Mapping:** Create a table mapping the raw CSV columns (e.g., best_bid_price, best_bid_qty) to Nautilus QuoteTick fields.

*   **Code Specification:** Provide the Python code structure for loading these files into the **Data Catalog**.

    *   Reference: Use nautilus_trader.persistence.wranglers or custom CSV loaders.

    *   Challenge: Address how to handle aggTrades (which are aggregate trade batches) versus the standard TradeTick. Suggest mapping is_buyer_maker to the AggressorSide.

#### **2.2 The Strategy Actor AlphaStrategy)**

Define the class structure for class AlphaStrategy(Strategy): inheriting from nautilus_trader.trading.strategy.Strategy [4].

*   **Initialization on_start):** Show how to subscribe to three different data streams simultaneously:

    *   self.subscribe_bars(BarType)

    *   self.subscribe_quote_ticks(InstrumentId)

    *   self.subscribe_trade_ticks(InstrumentId)

*   **State Management:** Define the internal buffers (e.g., Deqeue or CircularBuffer) required to calculate rolling VPIN and Volatility without re-calculating the entire history on every tick.

#### **2.3 Event-Driven Logic (The "Hot Loop")**

Provide detailed logic diagrams and pseudo-code for the event handlers:

1.  *on_bar(self, bar: Bar):** Trigger the **Yang-Zhang** update. Is this calculation heavy? If so, explain how to optimize it to avoid blocking the actor.

2.  *on_trade_tick(self, tick: TradeTick):**

    *   Logic: Update volume buckets. Check if the bucket matches the VPIN volume threshold.

    *   Logic: If VPIN > Threshold, emit a Signal.

3.  *on_quote_tick(self, quote: QuoteTick):**

    *   Logic: Update Bid-Ask Spread state.

    *   Crucial: Explain how to synchronize this with TradeTick data (e.g., do we use the timestamp ts_event to align data streams?).

#### **2.4 Signal Generation & Execution Routing**

*   **Signal Output:** Define a custom event class VolatilitySignal(Data) [4] that the strategy publishes to the MessageBus.

*   **Execution:** Describe how an **Execution Algorithm** (e.g., TWAP) subscribes to this signal.

    *   If VolatilitySignal.regime == HIGH_VARIANCE: Reduce TWAP interval (execute faster).

    *   If VolatilitySignal.regime == LOW_LIQUIDITY: Pause execution.

---

### **Output Format Rules**

1.  **Professional Tone:** Use specific Nautilus terminology InstrumentId, BarType, DataEngine).

2.  **Visuals:** Provide an ASCII sequence diagram showing the flow of a single TradeTick from CSV -> DataEngine -> Strategy.on_trade_tick -> VPIN Calculation -> Order Submission.

3.  **Math-to-Code Translation:** Explicitly comment on where the equations from Chapter 1 are implemented in the code structure.

**Begin generating Chapter 2 now.**
```
