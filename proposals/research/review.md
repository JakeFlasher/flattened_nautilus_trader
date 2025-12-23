## 0) TL;DR (8–15 bullets)

**Biggest strengths (3):**
- Strong **microstructure-aware framing** (latent efficient price + microstructure noise; explicit warning that “sample faster” can be pathological). ([nber.org](https://www.nber.org/papers/w9611?utm_source=openai))
- Good **event-driven system architecture direction** (CSV → typed objects → deterministic replay → Strategy/Exec/Risk separation), and it’s aligned with Nautilus’ `ts_event`/`ts_init` concepts in several places. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))
- Sensible inclusion of **flow toxicity as a risk control input** and explicit acknowledgement of VPIN’s critiques (that’s rare, and correct). ([kellogg.northwestern.edu](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/?utm_source=openai))

**Biggest weaknesses / fatal flaws (5–8):**
- **[Fatal] Alpha is not fully specified end-to-end**: Chapter 3 assumes a bounded normalized signal \(Z_t\in[-3,3]\), but Chapters 1–2 never define how \(Z_t\) is produced online (nor implement the Kalman filter promised in Chapter 1 §1.5). This breaks implementability.
- **[Fatal] Backtest realism is missing the dominant cost drivers** for perpetual futures: taker/maker fees, spread-crossing, slippage, latency, and (critically) **funding payments**. Without these, any performance conclusions are not research-grade. Funding is a core mechanism of perps. ([academy.binance.com](https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets?utm_source=openai))
- **[Major] Timestamp/order semantics are misstated**: Chapter 2 asserts “DataEngine merges streams by `ts_event`,” but Nautilus backtests order by **`ts_init`** and tie-break by **stream priority**—this matters for leakage and determinism. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))
- **[Major] Volatility estimator choice doesn’t match the HFT claim**: YZ-on-UTC-days from 1m OHLC is (a) slow to update, (b) not noise/jump robust, and (c) likely mis-scaled vs the decision interval used for trading. SOTA uses noise-robust realized measures and explicit volatility forecasting models. ([shephard.scholars.harvard.edu](https://shephard.scholars.harvard.edu/publications/designing-realised-kernels-measure-ex-post-variation-equity-prices-presence?utm_source=openai))
- **[Major] Unit/horizon consistency is not enforced**: the sizing law uses \(\mu_t/(\gamma\sigma_t^2)\), but \(\mu_t\) and \(\sigma_t\) must be defined on the *same* horizon; YZ is “session/day,” while decisions are seconds/minutes. ([nber.org](https://www.nber.org/papers/w8160?utm_source=openai))
- **[Major] VPIN implementation is plausible but uncalibrated and vulnerable to “mechanical” effects** (volume/intensity artifacts) unless you de-seasonalize and benchmark against simpler signed-flow measures. ([kellogg.northwestern.edu](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/?utm_source=openai))

**Highest-leverage fixes (3–5):**
- Define a **single, explicit online alpha pipeline**: features → model → \(Z_t\) (bounded) → \(\mu_t\) (horizon-consistent) → \(Q_\text{target}\); implement it in Chapter 2 (hot loop) and reference it in Chapter 3.
- Make the backtest **cost-complete and time-faithful** using Nautilus’ fill/slippage model + explicit fee + latency assumptions, and add a **funding treatment** (either add funding data or enforce “flat at funding times” as a conservative constraint). ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/backtesting/?utm_source=openai))
- Replace “YZ daily-only” with **noise/jump-robust realized measures** (midquote-based realized kernel / pre-averaging) and a **forecast model** (HAR-RV / Realized GARCH), then use volatility mainly for sizing/regime overlays. ([shephard.scholars.harvard.edu](https://shephard.scholars.harvard.edu/publications/designing-realised-kernels-measure-ex-post-variation-equity-prices-presence?utm_source=openai))


---

## 1) Grounded Outline of the Provided Proposal

### Chapter-by-chapter outline (short)

**Chapter 1 — Mathematical definition of alpha**
- Sets a latent efficient price \(X_t\) + observed noisy price \(Z_t=X_t+\varepsilon_t\) model; motivates realized variance and microstructure noise issues. ([nber.org](https://www.nber.org/papers/w9611?utm_source=openai))
- **Factor 1**: realized volatility using OHLC range estimators: Rogers–Satchell per bar, aggregated into a Yang–Zhang style estimator over rolling “sessions” (UTC days). ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aucp%3Ajnlbus%3Av%3A73%3Ay%3A2000%3Ai%3A3%3Ap%3A477-91?utm_source=openai))
- **Factor 2**: flow toxicity from `aggTrades` only: trade sign from `is_buyer_maker`, trade-flow imbalance (tOFI), VPIN in volume time.
- Proposes **Kalman filter** fusion into latent \(\alpha_t\), and mapping to target position \(\theta_t\propto \hat\alpha/(\gamma \sigma^2)\).

**Chapter 2 — NautilusTrader Architecture**
- Specifies CSV → ParquetDataCatalog ingestion mappings:
  - `bookTicker` → `QuoteTick`
  - `aggTrades` → `TradeTick` (aggressor side from `is_buyer_maker`)
  - `klines_1m` → `Bar` timestamped on close
- Implements event-driven strategy callbacks:
  - `on_bar`: RS/YZ updates
  - `on_trade_tick`: VPIN bucket fill/split
  - `on_quote_tick`: top-of-book liquidity state
- Publishes `VolatilitySignal` and adapts an execution algorithm (TWAP pacing/pausing) based on regimes.

**Chapter 3 — Risk Management & Portfolio Construction**
- Converts normalized signal \(Z_t\in[-3,3]\) into expected return via saturating \(\tanh\), then CRRA/Kelly fraction \(f_t\propto \mu_t/(\gamma\sigma_t^2)\).
- Converts fraction to target quantity \(Q^*(t)=f_t E_t/P_t\), applies optional vol-target cap, instrument quantization, and liquidity/spread gates.
- Adds risk engine reject layer + daily loss circuit breaker concept.

**Chapter 4 — Implementation Appendix**
- ETL glue code: deterministic streaming CSV parsing → typed Nautilus objects → Parquet catalog.
- Backtest wiring: define instrument, venue, run window (May 2023), attach strategy + exec algo, run and generate tearsheet.
- Defines standard metrics (Sharpe/Sortino/MDD) and tearsheet generation.

### Key claims & dependencies
- **Claim:** “Realized volatility + flow toxicity” yields tradable short-horizon alpha when fused online. *(Depends on a coherent \(Z_t\) definition and horizon-consistent model.)*
- **Claim:** 1m OHLC is “pragmatic HF” and can robustly produce volatility features. *(Depends on microstructure-noise handling and whether trade-price OHLC is adequate.)*
- **Claim:** VPIN from `aggTrades` is usable as toxicity gate. *(Depends on sign correctness, de-seasonalization, and benchmarking against simpler imbalance.)*
- **Claim:** Nautilus event-driven backtest with local CSV is deterministic and safe. *(Depends on correct `ts_init` semantics, tie-breaking, costs, and execution simulation configuration.)* ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/?utm_source=openai))

### Constraints (explicit restatement)
- Crypto **perpetual futures** (BTCUSDT perp), **24/7** market (no open/close).
- Volatility-based alpha focus (but proposal also uses order-flow toxicity).
- Backtesting focus: **offline CSV only** for trades (`aggTrades`), quotes (`bookTicker`), bars (1m klines).
- Must be implementable and reproducible with correct data contracts and event ordering.


---

## 2) SOTA Map (Web-Researched)

| Method/Idea | What problem it solves | Data requirements | Why it matters here | Key sources |
|---|---|---|---|---|
| Quadratic variation / realized volatility foundation | Justifies using high-frequency returns to estimate integrated variance | High-freq prices/returns | Your Chapter 1 foundation is correct; you should align features/targets to this theory | Andersen et al. (2001/2003) ([nber.org](https://www.nber.org/papers/w8160?utm_source=openai)) |
| Optimal sampling under microstructure noise | Explains why “max frequency” can be biased; motivates noise modeling | Tick data + noise model | Directly relevant to your “1m is not HF in theory” point; also impacts midquote sampling | Aït-Sahalia & Mykland (2003/2005) ([nber.org](https://www.nber.org/papers/w9611?utm_source=openai)) |
| Two-Scale / Multi-Scale Realized Volatility (TSRV/MSRV) | Noise-robust integrated volatility estimation | Tick/quote returns at multiple grids | Fits your constraint set (you have `bookTicker`), and is closer to SOTA than YZ-on-daily | Zhang (2004); Aït-Sahalia et al. (2011) ([arxiv.org](https://arxiv.org/abs/math/0411397?utm_source=openai)) |
| Pre-averaging estimator | Noise-robust IV with \(n^{-1/4}\) rates; also helps with dependent noise | Tick/quote data | Good candidate if you compute volatility from midquotes; robust alternative to 1m OHLC | Jacod et al. (pre-averaging) ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aaah%3Acreate%3A2007-43?utm_source=openai)) |
| Realized kernels | Noise-robust IV with dependent noise and irregular sampling support | Tick/quote data | Strong SOTA choice for a market like BTC perps with irregular event times | Barndorff-Nielsen et al. (2008) ([shephard.scholars.harvard.edu](https://shephard.scholars.harvard.edu/publications/designing-realised-kernels-measure-ex-post-variation-equity-prices-presence?utm_source=openai)) |
| Jump-robust realized variation (bipower/tripower, truncation) | Separates continuous vs jump variation; avoids jump contamination | Tick/quote returns | Crypto perps have jump/liquidation cascades; YZ alone doesn’t separate jump risk | Barndorff-Nielsen & Shephard (2004) ([academic.oup.com](https://academic.oup.com/jfec/article/2/1/1/960705?utm_source=openai)) |
| Range-based estimators (RS, Yang–Zhang) | Efficient volatility estimation from OHLC; drift robustness; “opening jump” handling | OHLC bars | Your current choice is defensible for coarse data, but not SOTA for microstructure-noise-rich tick settings | Yang & Zhang (2000) ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aucp%3Ajnlbus%3Av%3A73%3Ay%3A2000%3Ai%3A3%3Ap%3A477-91?utm_source=openai)) |
| HAR-RV volatility forecasting | Simple strong baseline for RV forecasting; captures long memory | Daily RV series (or multi-horizon RV) | If your “volatility arbitrage” claim is about forecasting, HAR should be a baseline | Corsi (2009) ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aoup%3Ajfinec%3Av%3A7%3Ay%3A2009%3Ai%3A2%3Ap%3A174-196?utm_source=openai)) |
| Realized GARCH | Joint model for returns + realized measures; improves over GARCH-only | Daily returns + realized measure | Strong SOTA for turning realized measures into a forecast \(\hat\sigma_{t+1}\) | Hansen, Huang & Shek (2012) ([pure.au.dk](https://pure.au.dk/portal/en/publications/realized-garch-a-joint-model-for-returns-and-realized-measures-of?utm_source=openai)) |
| Crypto-specific evidence: HAR works for BTC vol, volume adds info | Empirical support that HAR-type models fit BTC realized vol | BTC realized vol + volume | Supports your “volatility factor” framing, but also warns: returns often not predictable | Aalborg et al. (2019) ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aeee%3Afinlet%3Av%3A29%3Ay%3A2019%3Ai%3Ac%3Ap%3A255-265?utm_source=openai)) |
| Jump-robust realized measures improve BTC volatility forecasts | Shows jump-robust realized measures are important for BTC volatility modeling | BTC realized measures + returns | Directly argues you should add jump-robust realized measures if “volatility forecasting” is central | (NAJEF, 2020) ([sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S1062940820300620?utm_source=openai)) |
| Bitcoin volatility predictability: jumps + regimes | Vol regimes/jumps matter for forecasting; regime-switching helps | Realized measures + regime model inputs | Aligns with your “regime switching” story; suggests explicit regime model vs ad hoc thresholds | (FRL, 2022) ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S1544612322000162?utm_source=openai)) |
| OFI (order flow imbalance) explains short-horizon price moves better than volume | Motivates imbalance features; clarifies what “OFI” really is | L1 book events (best bid/ask queue changes) | Your tOFI is trade-only; you must label it clearly and benchmark; true OFI needs book deltas | Cont, Kukanov, Stoikov (2014) ([academic.oup.com](https://academic.oup.com/jfec/article/12/1/47/816163?utm_source=openai)) |
| VPIN (flow toxicity) | Measures toxicity in volume time | Signed trades + bucket volume | Relevant as risk gate; but must handle critiques and intensity artifacts | Easley, López de Prado, O’Hara (2012) ([academic.oup.com](https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929?utm_source=openai)) |
| VPIN critique: poor predictor; mechanical relation with trading intensity | Warns that VPIN can “work” due to mechanics not information | Signed trades + intensity | Your Chapter 1 notes this; you must hard-benchmark VPIN vs simpler indicators | Andersen & Bondarenko (2014) ([kellogg.northwestern.edu](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/?utm_source=openai)) |
| Perpetual futures microstructure (Binance perps intraday patterns; noise at 1m) | Crypto-specific intraday cycles; warns 1m can be noisy | Binance perp trades | Supports adding intraday seasonality controls and careful sampling frequency choices | (Borsa Istanbul Review, 2025) ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S2214845025001188?utm_source=openai)) |
| Funding rates in perps (mechanism) | Explains that funding is a systematic PnL component | Funding schedule + positions | If you hold over funding times, ignoring funding invalidates PnL attribution | Binance Academy (funding basics) ([academy.binance.com](https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets?utm_source=openai)) |
| Trade direction from `isBuyerMaker` | Trade aggressor inference without quotes | Trade feed with `isBuyerMaker` | Validates your sign convention (buyer maker ⇒ seller taker) | Binance dev glossary/forum ([developers.binance.com](https://developers.binance.com/docs/binance-spot-api-docs/faqs/spot_glossary?utm_source=openai)) |
| Optimal execution (AC model) | Execution cost/risk tradeoff; baseline for TWAP/VWAP reasoning | Price impact + volatility | If you claim “safe/optimal commands,” you need an execution cost model and/or realistic fill simulation | Almgren & Chriss (2001) ([risk.net](https://www.risk.net/journal-of-risk/technical-paper/2161150/optimal-execution-portfolio-transactions?utm_source=openai)) |
| Backtest data snooping controls | Controls false discoveries from parameter search | Multiple trials + backtests | Your sample window is short; you need PBO/Reality Check/DSR if tuning many knobs | White (2000) ([econometricsociety.org](https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping?utm_source=openai)); Bailey et al. (PBO) ([scholarworks.wmich.edu](https://scholarworks.wmich.edu/math_pubs/42/?utm_source=openai)); Bailey & López de Prado (DSR) ([econbiz.de](https://www.econbiz.de/Record/the-deflated-sharpe-ratio-correcting-for-selection-bias-backtest-overfitting-and-non-normality-bailey-david/10011433463?utm_source=openai)) |
| Time-faithful backtesting in Nautilus (`ts_init`, bar close convention, slippage model) | Prevents look-ahead; controls tie ordering; simulates fills with L1/L2 | Correct timestamps + fill model | Several parts of your Chapter 2–4 spec contradict or omit these mechanics | Nautilus docs ([nautilustrader.io](https://nautilustrader.io/docs/nightly/concepts/backtesting/?utm_source=openai)) |


---

## 3) Comprehensive Critique (Anchored to Chapters)

### 3.1 Conceptual/modeling issues

1) **Alpha definition is incomplete / misaligned across chapters**  
- **Severity:** Fatal  
- **Evidence:** Chapter 3 assumes an input “\(Z_t\in[-3,3]\)” and builds sizing on it (§3.0, §3.1), but Chapter 2 publishes only `VolatilitySignal(yz_var, vpin, spread_bps, regime)` and never constructs \(Z_t\). Chapter 1 proposes a Kalman filter (§1.5) but Chapter 2 does not implement it.  
- **Consequence if unfixed:** You cannot implement the strategy; results will be “strategy-shaped” but not what the research claims.  
- **How to detect:** Integration test: run backtest and log a time series of \((Z_t,\mu_t,\sigma_t,Q_\text{target})\). If \(Z_t\) is missing/constant/undefined, fail the build.

2) **“Volatility arbitrage” label is misleading for a futures-only directional strategy**  
- **Severity:** Major  
- **Evidence:** Title/Chapter 1 framing suggests “volatility arbitrage,” but the implementation trades BTCUSDT perps directionally (target position \(\theta_t\)), and volatility is mainly a feature/risk scaler.  
- **Consequence if unfixed:** Confused hypothesis → wrong evaluation criteria. Readers will expect variance-risk-premium/option-like logic, but you’re building a directional flow strategy with volatility regime gating.  
- **How to detect:** Write a one-paragraph hypothesis test: “If volatility rises, we expect ____ return sign over horizon ____ because ____.” If you can’t fill blanks without invoking order flow, it’s not volatility arbitrage.

3) **Volatility estimator choice doesn’t match the intended decision timescale**  
- **Severity:** Major  
- **Evidence:** Chapter 1/2 compute Yang–Zhang on rolling *days* (UTC sessions), updating only at day boundaries (Chapter 2 §2.3.1). Yet Chapter 2 discusses “HFT engine” decision-time and uses VPIN in trade-time.  
- **Consequence if unfixed:** Your \(\sigma_t\) will be stale for intraday risk control and any “volatility surprise” logic; sizing becomes inconsistent and regime detection lags.  
- **How to detect:** Compare volatility feature half-life vs trading cadence: compute autocorrelation of \(\sigma_t\) and average holding period; if volatility barely moves while trading frequently, you’re not using “real-time volatility.”

4) **Microstructure noise & jumps are acknowledged but not handled in the estimator**  
- **Severity:** Major  
- **Evidence:** Chapter 1 §1.2 correctly notes microstructure noise issues, but Factor 1 uses OHLC range estimators on trade-price bars without jump/noise robustness beyond aggregation. SOTA offers realized kernels / pre-averaging / multi-scale. ([shephard.scholars.harvard.edu](https://shephard.scholars.harvard.edu/publications/designing-realised-kernels-measure-ex-post-variation-equity-prices-presence?utm_source=openai))  
- **Consequence if unfixed:** Volatility features can be biased, regime thresholds become non-stationary, and “volatility surprise” can become a proxy for microstructure artifacts.  
- **How to detect:** Signature plot: compute realized volatility vs sampling interval (e.g., 1s, 5s, 30s, 1m, 5m) on the same day; if vol explodes as interval shrinks, you’re noise-dominated. ([nber.org](https://www.nber.org/papers/w9611?utm_source=openai))

5) **Trade-only “OFI” should not be called OFI without strong caveats**  
- **Severity:** Moderate  
- **Evidence:** Chapter 1 §1.4.2 admits classical OFI needs L1 queue changes, but then defines “tOFI.” That’s fine, but Chapter 1 still leans on OFI literature for justification. True OFI’s robustness claims don’t transfer automatically. ([academic.oup.com](https://academic.oup.com/jfec/article/12/1/47/816163?utm_source=openai))  
- **Consequence if unfixed:** Overclaims: you may attribute predictive power to OFI theory, while actually using signed volume imbalance (which behaves differently).  
- **How to detect:** Benchmark: regress short-horizon returns on tOFI vs (a) true OFI if you later add L1 deltas, (b) signed volume only, and check incremental \(R^2\)/IC.

---

### 3.2 Statistical/evaluation issues (leakage, microstructure bias, overfitting, stationarity)

1) **Cost omission makes any Sharpe/edge claims non-credible for HFT**  
- **Severity:** Fatal  
- **Evidence:** Chapters 2–4 show execution logic but do not specify exchange fees, spread-crossing, or realistic slippage/latency assumptions. Nautilus provides explicit fill/slippage mechanisms for L1/bar data, but your spec doesn’t configure them. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/backtesting/?utm_source=openai))  
- **Consequence if unfixed:** Backtest PnL will be structurally inflated; you can easily “discover” false edges.  
- **How to detect:** Sensitivity test: rerun with taker fee = 0, 2 bps, 5 bps and slippage = 0, 1 tick (probabilistic). If strategy flips from profitable to unprofitable, costs dominate and must be modeled.

2) **Funding is ignored (perps-specific PnL component)**  
- **Severity:** Fatal  
- **Evidence:** No chapter models funding transfers; Chapter 1 even highlights funding regime switching in the threat model, but it is not in the backtest math or accounting. Funding is central to perp pricing and holding costs. ([academy.binance.com](https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets?utm_source=openai))  
- **Consequence if unfixed:** PnL attribution and risk are wrong, especially around funding timestamps; strategy may “work” only because you implicitly assume free carry.  
- **How to detect:** If your realized holding periods overlap common funding times, you must compute hypothetical funding PnL bounds; if the bounds are comparable to strategy edge, current results are invalid.

3) **Small sample window + parameter tuning risk**  
- **Severity:** Major  
- **Evidence:** Chapter 4 uses May 16–31, 2023 (~16 days). Many parameters exist (VPIN bucket size/window/threshold, spread gates, YZ window, TWAP pacing).  
- **Consequence if unfixed:** High probability of backtest overfitting / selection bias. SOTA uses data-snooping-aware evaluation (Reality Check, PBO, DSR). ([econometricsociety.org](https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping?utm_source=openai))  
- **How to detect:** Track “# trials” and compute PBO/Reality Check on your configuration search; if you can’t enumerate trials, assume overfit. ([scholarworks.wmich.edu](https://scholarworks.wmich.edu/math_pubs/42/?utm_source=openai))

4) **Horizon/units mismatch between \(\mu\) and \(\sigma\)**  
- **Severity:** Major  
- **Evidence:** Chapter 3 defines \(\mu_t\) as expected return “per decision step,” but \(\sigma_t\) is derived from YZ computed over sessions/days (Chapter 1 §1.3.2, Chapter 2 §2.3.1).  
- **Consequence if unfixed:** Kelly/CRRA sizing is dimensionally inconsistent; leverage can be too high/low in ways that look “good” in backtest but fail live.  
- **How to detect:** Unit test: annotate every variable with “per second / per minute / per day”. Refuse to run if mismatched.

---

### 3.3 Data/engineering issues (timestamping, stream joins, determinism, survivorship)

1) **Nautilus event ordering is misstated; tie-breaking is underspecified**  
- **Severity:** Major  
- **Evidence:** Chapter 2 claims DataEngine merges by `ts_event` and that ties may vary; but Nautilus backtesting orders by `ts_init` and tie-breaks by stream priority (`append_data`). ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))  
- **Consequence if unfixed:** You can accidentally introduce look-ahead or non-determinism when bars/trades/quotes share timestamps; results change with ingestion order.  
- **How to detect:** Determinism test: run the same backtest twice with shuffled file ingestion order; if results differ, ordering assumptions are wrong.

2) **Bar timestamping is “right” but must be aligned with Nautilus’ execution expectations**  
- **Severity:** Moderate  
- **Evidence:** You timestamp bars on close time (good), and Nautilus explicitly warns that `ts_init` must represent bar close for bar execution simulation. ([nautilustrader.io](https://nautilustrader.io/docs/nightly/concepts/backtesting/?utm_source=openai))  
- **Consequence if unfixed:** If anyone later changes timestamps (open vs close) without `ts_init_delta`, you can get look-ahead fills.  
- **How to detect:** Add a guard: verify each bar’s `ts_init` equals its close boundary; fail ingestion otherwise.

3) **Aggressor-side mapping is correct in principle, but you must lock it to a verified definition**  
- **Severity:** Moderate  
- **Evidence:** Chapter 1/2 mapping “`is_buyer_maker=True` ⇒ seller-initiated (aggressor SELL)” matches Binance definitions (“buyer is maker”). ([developers.binance.com](https://developers.binance.com/docs/binance-spot-api-docs/faqs/spot_glossary?utm_source=openai))  
- **Consequence if unfixed:** If dataset semantics differ (spot vs futures export quirks), VPIN sign flips and strategy inverts.  
- **How to detect:** Cross-check against contemporaneous midquote changes: buyer-initiated trades should correlate with upward mid moves at very short horizons (weakly but directionally).

4) **Data quality controls are missing (duplicates, monotonicity, gaps)**  
- **Severity:** Moderate  
- **Evidence:** Chapter 2/4 parse defensively, but there’s no QC spec for monotonic timestamps, duplicate IDs, cross-file overlaps, or missing minutes.  
- **Consequence if unfixed:** Feature leakage (e.g., out-of-order ticks), incorrect VPIN bucket accounting, and spurious regime triggers.  
- **How to detect:** QC report per day:
  - % non-monotonic `ts_event`
  - duplicate `agg_trade_id` / `update_id`
  - missing 1m bars
  - fraction of quote ticks with crossed markets (ask < bid)

---

### 3.4 Risk/ops issues (position sizing stability, circuit breakers, stress scenarios)

1) **Perpetual futures margin/liquidation mechanics are not modeled**  
- **Severity:** Major  
- **Evidence:** Chapter 3 sizes on equity/notional, but does not specify leverage caps, maintenance margin, liquidation triggers, or mark price vs last price accounting. Funding is also absent. ([academy.binance.com](https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets?utm_source=openai))  
- **Consequence if unfixed:** Strategy can appear “safe” while actually being liquidation-prone live.  
- **How to detect:** Stress test: apply a synthetic 5–10% adverse jump and see if sizing would breach plausible leverage/margin limits; if yes, risk model incomplete.

2) **Circuit breakers exist conceptually but are not integrated into execution policy**  
- **Severity:** Moderate  
- **Evidence:** Chapter 3 mentions daily loss circuit breaker and REDUCING mode, but Chapter 2 exec algo can continue spawning child orders via timers.  
- **Consequence if unfixed:** You can keep trading into a stop condition due to asynchronous timers.  
- **How to detect:** Unit test: force `TradingState.REDUCING` mid-TWAP and ensure no new child orders spawn and existing ones cancel/expire.

3) **Liquidity gating relies on spread/top size but ignores quote staleness**  
- **Severity:** Moderate  
- **Evidence:** Chapter 2 §2.3.3 uses spread and top size thresholds, but not staleness relative to decision time. Nautilus supports latency analysis via `ts_init-ts_event`. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))  
- **Consequence if unfixed:** You may trade using stale quotes during feed gaps, which is dangerous in crypto spikes.  
- **How to detect:** Track quote age distribution; halt trading when quote age exceeds threshold (e.g., >500ms in “HFT mode”, >5s in “minute mode”).

---

## 4) Fixes & Redesign (Research-Backed)

Below are **minimum viable** corrections to make the proposal research-grade and implementable, followed by better (more effort) versions.

### A) [Fatal] Define \(Z_t\) and the alpha model end-to-end (Ch1→Ch2→Ch3 integration)

**Essential fix (minimum viable): “z-score ensemble” with explicit horizon**
- Define a **decision interval** \(\Delta\) (assumption: \(\Delta=1\text{ minute}\) initially, since you already have 1m bars and want stability; you can later move to seconds).
- Define features available at decision time \(t\):
  - \(z^{(\text{vol})}_t\): standardized volatility surprise (already sketched in Ch1).
  - \(z^{(\text{flow})}_t\): standardized signed-flow imbalance (e.g., \(\widetilde{\text{tOFI}}_{t,\Delta}\)) gated by VPIN and liquidity state.
- Construct the canonical bounded signal:
  \[
  Z_t = \mathrm{clip}\big(w_1 z^{(\text{vol})}_t + w_2 z^{(\text{flow})}_t,\,-3,\,+3\big)
  \]
- In Chapter 2 implement `self.alpha_z = Z_t` updated on a **timer** or `on_bar`.

**Better fix:** implement **explicit online regression or Kalman filter** with estimable parameters
- Keep Chapter 1’s state-space idea but make it *operational*:
  - Define a forecast target: next-interval midquote return \(r_{t\to t+\Delta}=\log(m_{t+\Delta}/m_t)\).
  - Use a linear observation model for returns (simpler than “features observe alpha”):
    \[
    r_{k+1} = \beta^\top f_k + \epsilon_{k+1}
    \]
  - Estimate \(\beta\) with rolling ridge regression or recursive least squares.
- This is closer to research-grade than a Kalman filter with unspecified \(H,R,Q\).

**Tradeoffs & failure modes**
- Ensemble is robust but can be arbitrary; regression is principled but can overfit and leak if labels overlap.
- If you later go sub-minute, you must handle label overlap with embargo/purging and microstructure effects.

**Step-by-step workflow (implementable)**
```python
# Decision loop at fixed interval Δ
f_vol  = robust_zscore(vol_measure_t, window=Wv)
f_flow = robust_zscore(signed_flow_t, window=Wf) * gate(vpin_t, spread_t, quote_age_t)

Z_t = clip(w1*f_vol + w2*f_flow, -3, +3)
mu_t = kappa * tanh(Z_t / z_sat)              # now μ_t is per-Δ expected return
sigma_t = sigma_forecast_for_horizon_Δ()      # MUST match Δ
Q_target = position_sizing(mu_t, sigma_t, equity, midprice, leverage_caps, quantization)
```

---

### B) [Fatal] Add **cost-complete** backtesting: fees, spread, slippage, latency, funding

**Essential fix (minimum viable, under your data constraints)**
1) **Fees**
   - Parameterize maker/taker bps (don’t hardcode a VIP tier).
   - Apply to each fill in backtest accounting.

2) **Spread & slippage**
   - Ensure fills occur at bid/ask (market orders cross the spread).
   - In Nautilus configure a `FillModel` with non-zero `prob_slippage` for L1/bar backtests. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/backtesting/?utm_source=openai))

3) **Latency**
   - Add a deterministic latency \(\ell\) (e.g., 50ms, 200ms, 500ms) by setting `ts_init = ts_event + ℓ` for *orders* (or by delaying order submission logic in strategy).
   - Keep market data `ts_init=ts_event` if you want, but be explicit: “zero feed latency assumption.”

4) **Funding**
   - Since your CSV set doesn’t include funding rates, do **one** of:
     - **Conservative constraint:** enforce **flat/neutral at funding timestamps** (reduce-only) so funding PnL is ~0 by design.
     - **Add a new local dataset** of funding rates and apply actual funding transfers.
   - Funding is fundamental to perps; Binance explains the mechanism and that rates are periodic. ([academy.binance.com](https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets?utm_source=openai))

**Better fix (production-grade)**
- Add **mark price** time series (or index) and model liquidation/margin based on mark, not last trade.
- Use L2/L3 order book if you can obtain it; Nautilus’ fill model is much more realistic with depth. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/backtesting/?utm_source=openai))

**Tradeoffs & failure modes**
- Conservative “flat at funding” reduces strategy flexibility; but it’s the only honest fix without funding data.
- Adding funding data increases ETL complexity but is required for a credible perp strategy.

**Concrete tests**
- “Cost dominance test”: PnL must be positive under realistic fee + slippage ranges, not just at 0 costs.
- “Funding exposure test”: fraction of time holding across funding times must be ≈0 if you choose the “flat at funding” policy.

---

### C) [Major] Correct event ordering semantics and prevent timestamp-induced leakage

**Essential fix**
- Update the spec: in Nautilus backtests, **chronological processing is by `ts_init`**, not `ts_event`; tie order is controlled by stream priority (`append_data`). ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))
- Decide and document a tie-break rule that matches your desired causality:
  - Example: process `QuoteTick` before `TradeTick` when same timestamp (so spread is known before fills).
- Implement by setting stream priorities when adding data (or by shifting `ts_init` by 1ns for one stream).

**Better fix**
- Use exchange sequence IDs to enforce ordering inside identical timestamps:
  - `bookTicker.update_id` and `agg_trade_id` can be used for auditing and possibly ordering.

**Failure modes**
- Artificially forcing quotes before trades can hide real-world race conditions; but it’s better than nondeterminism.
- If you don’t control ties, backtest results can change with ingestion order.

**Detection tests**
- Determinism test: shuffle file ingestion order; results must match bit-for-bit.
- Look-ahead test: verify that no strategy decision uses data with `ts_event > decision_ts_event`.

---

### D) [Major] Upgrade volatility measurement to noise/jump-robust realized measures + forecasting

**Essential fix (minimum viable)**
- Keep your current YZ/RS pipeline but:
  - Move from “daily boundary only” to **rolling intraday updates** (e.g., update every minute using last N minutes of RS contributions).
  - Explicitly scale \(\sigma\) to the trading horizon \(\Delta\).
- Add a baseline forecast model: **HAR-RV** on log realized vol (even if crude). ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aoup%3Ajfinec%3Av%3A7%3Ay%3A2009%3Ai%3A2%3Ap%3A174-196?utm_source=openai))

**Better fix (recommended SOTA under your available data)**
- Compute volatility from **midquotes** (bookTicker):
  - \(m_t=(b_t+a_t)/2\), returns on a regular grid (e.g., 1s or 5s).
- Use **realized kernel** or **pre-averaging** to handle noise. ([shephard.scholars.harvard.edu](https://shephard.scholars.harvard.edu/publications/designing-realised-kernels-measure-ex-post-variation-equity-prices-presence?utm_source=openai))
- Add a forecast model:
  - HAR-RV baseline (Corsi). ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aoup%3Ajfinec%3Av%3A7%3Ay%3A2009%3Ai%3A2%3Ap%3A174-196?utm_source=openai))
  - Realized GARCH as stronger baseline if you want parametric structure. ([pure.au.dk](https://pure.au.dk/portal/en/publications/realized-garch-a-joint-model-for-returns-and-realized-measures-of?utm_source=openai))
- Add jump-robust realized measure (bipower/tripower) if jumps dominate; crypto evidence suggests jump-robust realized measures can improve volatility forecasts. ([academic.oup.com](https://academic.oup.com/jfec/article/2/1/1/960705?utm_source=openai))

**Failure modes**
- Midquote sampling at too high frequency can still be noisy; you must pick a grid and validate with a signature plot. ([nber.org](https://www.nber.org/papers/w9611?utm_source=openai))
- Regime changes in crypto are strong; static parameters drift.

**Implementation sketch**
```python
# Build midquote grid (e.g., every 5s)
m_t = last_midquote_at_or_before(t)

# Realized measure (conceptual)
rv_t = realized_kernel(m_grid_returns)  # or pre_averaging(micro_returns)

# Forecast
log_rv_fore = HAR(log_rv_lags).predict()
sigma_fore = sqrt(exp(log_rv_fore))     # horizon-consistent
```

---

### E) [Major] Fix VPIN usage: calibration, de-seasonalization, benchmarking

**Essential fix**
- Treat VPIN strictly as a **risk gate**, not an alpha predictor, unless you prove incremental predictive power (you already gesture at this, but you must enforce it).
- Add **seasonality adjustment**:
  - Compute time-of-day baseline VPIN (or baseline imbalance) and z-score relative to that.
- Benchmark against simpler metrics:
  - signed volume imbalance, tOFI alone, and spread/quote-based stress.
  - This is aligned with the VPIN critique that predictive content can be mechanical. ([kellogg.northwestern.edu](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/?utm_source=openai))

**Better fix**
- Use a multivariate “stress index” combining:
  - VPIN,
  - spread z-score,
  - quote age,
  - trade intensity,
  and evaluate in an event study around high-stress windows.

**Detection tests**
- “Mechanical correlation test”: correlate VPIN with trade count/volume; if extremely high, you must control for intensity. ([kellogg.northwestern.edu](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/?utm_source=openai))
- “Benchmark test”: VPIN must beat a signed-flow z-score baseline at forecasting short-horizon volatility/stress.

---

### F) [Major] Make sizing futures-correct (margin, leverage caps, horizon consistency)

**Essential fix**
- Define:
  - exposure = notional / equity (leverage proxy),
  - hard leverage cap \(L_\max\),
  - max position \(Q_\max = L_\max E / P\).
- Ensure \(\mu_t\) and \(\sigma_t\) are both per-\(\Delta\) before using \(f_t=\mu_t/(\gamma\sigma_t^2)\). ([nber.org](https://www.nber.org/papers/w8160?utm_source=openai))
- Add “reduce-only mode” enforcement into execution: if REDUCING, cancel TWAP timers and only trade toward flat.

**Better fix**
- Add liquidation-aware sizing:
  - approximate maintenance margin,
  - stress move scenarios (e.g., 3–5 sigma + jump),
  - ensure liquidation probability under stress < threshold.

**Failure modes**
- If you rely on “equity” but the backtest doesn’t model mark-to-market/funding, equity is ill-defined.
- Kelly sizing is fragile under estimation error; clipping helps but does not solve unit mismatch.

---

## 5) Validation & Experiment Plan (Make it Research-Grade)

### Required backtest realism checklist (minimum bar for credibility)
- **Market data timing**
  - Validate `ts_init` ordering and stream priority; deterministic ties. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/?utm_source=openai))
  - Bars: `ts_init` must be close time if bar execution is used. ([nautilustrader.io](https://nautilustrader.io/docs/nightly/concepts/backtesting/?utm_source=openai))
- **Execution simulation**
  - Use quotes/trades for execution when possible; configure Nautilus fill model (slippage, queue probability).
  - Explicit spread handling: buys at ask, sells at bid. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/backtesting/?utm_source=openai))
- **Fees**
  - Maker/taker fees parameterized; sensitivity sweep across plausible tiers.
- **Latency**
  - At least one latency scenario (0ms, 50–200ms, 500ms) and show robustness.
- **Perps-specific**
  - Funding modeled or neutralized (“flat at funding times”). ([academy.binance.com](https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets?utm_source=openai))
  - Leverage cap + liquidation-avoidance policy.
- **Data QC**
  - Monotonicity, duplicates, gaps; daily QC report.

### Ablations (prove causality)
- Remove VPIN gate → does drawdown worsen in stress windows?
- Remove volatility feature (use constant \(\sigma\) sizing) → does risk-adjusted performance degrade?
- Remove signed-flow feature → does any alpha remain (or does strategy become only “risk timing”)?
- Replace YZ with a simple RV baseline → does performance persist?

### Robustness tests
- **Regime segmentation** (crypto is 24/7 and session cycles matter):
  - Asia/Europe/US hours; weekends vs weekdays.
- **Stress periods**:
  - Highest realized vol days; liquidation cascade proxies (rapid spread widening + high trade intensity).
- **Parameter stability**:
  - Walk-forward calibration; rolling re-fit.
  - Report sensitivity surfaces for VPIN bucket volume, threshold, and execution pacing.
- **Stationarity checks**
  - Compare feature distributions across months/quarters; if drift is large, require adaptive normalization. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S2214845025001188?utm_source=openai))

### Metrics beyond Sharpe
- Tail risk: max drawdown, expected shortfall, worst 1h/1d PnL
- Turnover, trade count, average holding period, % time in market
- Cost breakdown: gross PnL vs fees vs spread vs slippage vs funding
- Capacity proxy: PnL per unit notional vs estimated impact proxies (spread, depth)

---

## 6) Chapter-by-Chapter Revision Plan (Patch List)

### Chapter 1 — Mathematical definition of alpha

**What to change**
- Clarify that the strategy is **directional** and volatility is primarily a **risk/regime** input unless you explicitly define a volatility→return mechanism.
- Add an explicit definition of **decision interval \(\Delta\)** and enforce **horizon consistency** for \(\mu_t\) and \(\sigma_t\).
- Add at least one **noise-robust realized measure** option using bookTicker midquotes (realized kernel / pre-averaging) as “SOTA Track,” keeping YZ as “fallback track.”
- Tighten citations: replace low-authority links with primary sources for realized measures and microstructure noise.

**Where (section references)**
- Chapter 1 §1.1 (time scales): add \(\Delta\) definition.
- Chapter 1 §1.3 (Factor 1): add “SOTA track: midquote realized kernel / pre-averaging.”
- Chapter 1 §1.5: explicitly define \(Z_t\) (or \(\alpha_t\)) output and its online computation.

**Proposed replacement text (critical paragraphs only)**

1) **Replace the “volatility arbitrage” positioning (front of Chapter 1)**  
> *Replacement text:*  
> “This system is a **directional perpetual-futures strategy** whose edge, if any, arises from **short-horizon order-flow imbalance conditioned on volatility and liquidity regimes**. Volatility features are used both as (i) predictive covariates and (ii) **risk-scaling/regime controls**. We do not claim an options-style variance risk premium ‘volatility arbitrage’ because we do not trade convexity; instead, we test whether volatility regimes modulate the predictability and execution risk of directional trades.”

2) **Add a strict horizon contract (end of §1.1 or start of §1.5)**  
> *Replacement text:*  
> “We fix a decision interval \(\Delta\) (initially \(\Delta=60\) seconds for stability). All quantities used in sizing must share this horizon: \(\mu_t := \mathbb{E}[r_{t\to t+\Delta}]\) and \(\sigma_t^2 := \mathrm{Var}(r_{t\to t+\Delta})\). Any daily/session volatility estimate must be explicitly scaled or replaced by a \(\Delta\)-horizon estimator.”

**Risks introduced**
- More complexity (two volatility pipelines).
- Forces you to confront whether volatility is predictive or only a risk control.

**Acceptance criteria**
- A single page in Chapter 1 defines \(Z_t\), \(\mu_t\), \(\sigma_t\), and \(\Delta\) with units and how each is computed without look-ahead.
- At least one volatility estimator is noise-robust and referenced to primary literature. ([shephard.scholars.harvard.edu](https://shephard.scholars.harvard.edu/publications/designing-realised-kernels-measure-ex-post-variation-equity-prices-presence?utm_source=openai))

---

### Chapter 2 — NautilusTrader Architecture

**What to change**
- Correct the event ordering statement: backtest ordering is by **`ts_init`**, not `ts_event`; tie-breaking is via stream priority. ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/?utm_source=openai))
- Add explicit venue/backtest configuration for execution realism:
  - bar vs trade execution, fill model, slippage probability, etc. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/backtesting/?utm_source=openai))
- Implement the missing **alpha signal production** (\(Z_t\)) and store it for Chapter 3 sizing.
- Fix the `VolatilitySignal` publishing logic bug (`_maybe_publish_vol_signal` currently no-ops when `ts_event_ns is None`).

**Where**
- Chapter 2 §2.1.2 (timestamp contracts): add `ts_init` semantics and tie policy.
- Chapter 2 §2.3.3 (synchronization): replace “do not attempt to join streams” with “set priorities + deterministic tie policy.”
- Chapter 2 §2.4.1–2.4.2 (signals/execution): integrate \(Z_t\) and risk state; ensure exec algo cancels/pauses cleanly.

**Proposed replacement text (critical paragraph)**
- **Replace “DataEngine merges streams by ts_event” (appears in diagrams/wording)**  
> *Replacement text:*  
> “In Nautilus backtests, data from multiple streams is merged and processed in strict chronological order using **`ts_init`** (nanoseconds). When multiple data points share the same `ts_init`, the processing order is deterministic and controlled by **stream priority** (e.g., `append_data=False` yields higher priority). We therefore treat `ts_event` as ‘exchange event time’ and `ts_init` as the deterministic simulation time used for causal processing and execution.” ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/?utm_source=openai))

**Risks introduced**
- You will need to decide a tie policy; no choice is perfect, but nondeterminism is worse.
- Tight coupling to Nautilus semantics (good for correctness).

**Acceptance criteria**
- Two identical backtest runs produce identical results (same fills, PnL, metrics).
- Logs show deterministic ordering for ties (quotes vs trades vs bars).

---

### Chapter 3 — Risk Management & Portfolio Construction

**What to change**
- Make sizing **futures-correct**: define leverage/notional, caps, and (at least approximate) margin/liquidation constraints.
- Enforce horizon consistency: \(\mu_t\) and \(\sigma_t\) must both be per-\(\Delta\).
- Integrate funding handling:
  - either model funding PnL,
  - or enforce “flat at funding times” and document why. ([academy.binance.com](https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets?utm_source=openai))
- Make REDUCING mode cancel/disable execution timers (no lingering TWAP child orders).

**Where**
- Chapter 3 §3.1.2–3.1.3 (sizing law, quantity conversion): add leverage cap and horizon scaling.
- Chapter 3 §3.2 (risk engine): add perps-specific constraints and explicit fee/funding assumptions.
- Chapter 3 §3.3 (guardrails): add quote staleness gate using `ts_init-ts_event`. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))

**Proposed replacement text (critical paragraph)**
> *Replacement text:*  
> “Because BTCUSDT perpetuals are margined instruments, we treat \(f_t\) as a **target leverage fraction** (not a cash allocation). We enforce a hard leverage cap \(L_\max\), so that \(|Q_\text{target}(t)| \le (L_\max E_t)/P_t\). Additionally, all sizing inputs must share the same decision horizon \(\Delta\): \(\mu_t=\mathbb{E}[r_{t\to t+\Delta}]\) and \(\sigma_t^2=\mathrm{Var}(r_{t\to t+\Delta})\). If we cannot compute \(\sigma_t\) on the \(\Delta\) horizon reliably, the system defaults to REDUCING/flat.”

**Risks introduced**
- Conservative caps may reduce backtest returns but increase real-world plausibility.
- Requires explicit assumptions (good).

**Acceptance criteria**
- Strategy never exceeds leverage cap in backtest.
- When quotes are stale or volatility invalid, target goes to 0 and no new orders are spawned.

---

### Chapter 4 — Implementation Appendix

**What to change**
- Add a **data QC stage** after ETL (monotonicity, duplicates, gaps, crossed quotes).
- Add explicit **Nautilus backtest configuration** for:
  - bar/trade/quote execution modes,
  - fill model parameters (`prob_slippage`, `prob_fill_on_limit`),
  - and tie-breaking priorities. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/backtesting/?utm_source=openai))
- Expand the experiment window and methodology:
  - If only May 2023 is available, label results as **smoke test only**.
  - Add walk-forward or multi-month evaluation plan.

**Where**
- Chapter 4 §4.1 (ETL): add QC report.
- Chapter 4 §4.2.4 (BacktestNode config): add venue config details and fill model.
- Chapter 4 §4.4 (metrics): add cost decomposition and turnover/capacity proxies.

**Proposed replacement text (critical paragraph)**
> *Replacement text:*  
> “Backtest validity depends on cost and fill realism. We configure Nautilus’ backtest venue with (i) the appropriate execution mode (trade/quote/bar execution), (ii) an explicit FillModel including L1 slippage probability, and (iii) deterministic stream priority for timestamp ties. We report PnL decomposed into gross edge, fees, spread/slippage, and (if modeled) funding.” ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/backtesting/?utm_source=openai))

**Risks introduced**
- More parameters; must be fixed before running experiments to avoid p-hacking.

**Acceptance criteria**
- Backtest run script fully specifies fill model + fees + funding handling.
- Produces reproducible tearsheet **and** a cost breakdown report.

---

## 7) References (Web Sources)

### Realized volatility, microstructure noise, and noise/jump-robust estimators
- Andersen, T.G., Bollerslev, T., Diebold, F.X., Labys, P. (2001/2003). *Modeling and Forecasting Realized Volatility*. NBER WP 8160 / Econometrica. ([nber.org](https://www.nber.org/papers/w8160?utm_source=openai))  
- Aït-Sahalia, Y., Mykland, P.A. (2003/2005). *How Often to Sample a Continuous-Time Process in the Presence of Market Microstructure Noise*. NBER WP 9611 / Review of Financial Studies. ([nber.org](https://www.nber.org/papers/w9611?utm_source=openai))  
- Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., Shephard, N. (2008). *Designing Realised Kernels to Measure the Ex-Post Variation of Equity Prices in the Presence of Noise*. Econometrica. ([shephard.scholars.harvard.edu](https://shephard.scholars.harvard.edu/publications/designing-realised-kernels-measure-ex-post-variation-equity-prices-presence?utm_source=openai))  
- Jacod, J., Li, Y., Mykland, P.A., Podolskij, M., Vetter, M. (2007). *Microstructure Noise in the Continuous Case: The Pre-Averaging Approach*. ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aaah%3Acreate%3A2007-43?utm_source=openai))  
- Zhang, L. (2004). *Efficient Estimation of Stochastic Volatility Using Noisy Observations: A Multi-Scale Approach*. arXiv. ([arxiv.org](https://arxiv.org/abs/math/0411397?utm_source=openai))  
- Aït-Sahalia, Y., Mykland, P.A., Zhang, L. (2011). *Ultra High Frequency Volatility Estimation with Dependent Microstructure Noise*. Journal of Econometrics (listing/version). ([ideas.repec.org](https://ideas.repec.org/a/eee/econom/v160y2011i1p160-175.html?utm_source=openai))  
- Barndorff-Nielsen, O.E., Shephard, N. (2004). *Power and Bipower Variation with Stochastic Volatility and Jumps*. Journal of Financial Econometrics. ([academic.oup.com](https://academic.oup.com/jfec/article/2/1/1/960705?utm_source=openai))  
- Yang, D., Zhang, Q. (2000). *Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices*. The Journal of Business. ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aucp%3Ajnlbus%3Av%3A73%3Ay%3A2000%3Ai%3A3%3Ap%3A477-91?utm_source=openai))  

### Volatility forecasting / regime models (incl. crypto evidence)
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility (HAR-RV)*. Journal of Financial Econometrics. ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aoup%3Ajfinec%3Av%3A7%3Ay%3A2009%3Ai%3A2%3Ap%3A174-196?utm_source=openai))  
- Hansen, P.R., Huang, Z., Shek, H.H. (2012). *Realized GARCH: A Joint Model for Returns and Realized Measures of Volatility*. Journal of Applied Econometrics. ([pure.au.dk](https://pure.au.dk/portal/en/publications/realized-garch-a-joint-model-for-returns-and-realized-measures-of?utm_source=openai))  
- Aalborg, H.A., Molnár, P., de Vries, J.E. (2019). *What can explain the price, volatility and trading volume of Bitcoin?* Finance Research Letters. ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aeee%3Afinlet%3Av%3A29%3Ay%3A2019%3Ai%3Ac%3Ap%3A255-265?utm_source=openai))  
- (2020). *Improving the realized GARCH’s volatility forecast for Bitcoin with jump-robust estimators*. The North American Journal of Economics and Finance. ([sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S1062940820300620?utm_source=openai))  
- (2022). *Bitcoin volatility predictability – The role of jumps and regimes*. Finance Research Letters. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S1544612322000162?utm_source=openai))  

### Order flow, toxicity (VPIN), and critiques
- Cont, R., Kukanov, A., Stoikov, S. (2014). *The Price Impact of Order Book Events*. Journal of Financial Econometrics / arXiv. ([academic.oup.com](https://academic.oup.com/jfec/article/12/1/47/816163?utm_source=openai))  
- Easley, D., López de Prado, M., O’Hara, M. (2012). *Flow Toxicity and Liquidity in a High-frequency World*. Review of Financial Studies. ([academic.oup.com](https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929?utm_source=openai))  
- Andersen, T.G., Bondarenko, O. (2014). *VPIN and the Flash Crash*. Journal of Financial Markets (Kellogg listing). ([kellogg.northwestern.edu](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/?utm_source=openai))  
- Easley, D., López de Prado, M., O’Hara, M. (2014). *VPIN and the Flash Crash: A rejoinder*. Journal of Financial Markets. ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aeee%3Afinmar%3Av%3A17%3Ay%3A2014%3Ai%3Ac%3Ap%3A47-52?utm_source=openai))  

### Execution and market impact
- Almgren, R., Chriss, N. (2001). *Optimal execution of portfolio transactions*. Journal of Risk (listing). ([risk.net](https://www.risk.net/journal-of-risk/technical-paper/2161150/optimal-execution-portfolio-transactions?utm_source=openai))  
- Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica (listing). ([econometricsociety.org](https://www.econometricsociety.org/publications/econometrica/browse/1985/11/01/continuous-auctions-and-insider-trading?utm_source=openai))  
- Obizhaeva, A., Wang, J. (2013). *Optimal trading strategy and supply/demand dynamics*. Journal of Financial Markets (listing). ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Aeee%3Afinmar%3Av%3A16%3Ay%3A2013%3Ai%3A1%3Ap%3A1-32?utm_source=openai))  

### Backtesting methodology (data snooping / overfitting controls)
- White, H. (2000). *A Reality Check for Data Snooping*. Econometrica. ([econometricsociety.org](https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping?utm_source=openai))  
- Bailey, D.H., Borwein, J., López de Prado, M., Zhu, Q.J. (2017). *The Probability of Backtest Overfitting*. Journal of Computational Finance (listing). ([scholarworks.wmich.edu](https://scholarworks.wmich.edu/math_pubs/42/?utm_source=openai))  
- Bailey, D.H., López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management (listing). ([econbiz.de](https://www.econbiz.de/Record/the-deflated-sharpe-ratio-correcting-for-selection-bias-backtest-overfitting-and-non-normality-bailey-david/10011433463?utm_source=openai))  
- Harvey, C.R., Liu, Y., Zhu, H. (2014). *. . . and the Cross-Section of Expected Returns* (multiple testing framework). NBER WP 20592. ([nber.org](https://www.nber.org/papers/w20592?utm_source=openai))  

### Crypto perps mechanics and Binance data semantics
- Binance Open Platform (developers) glossary: definition of `isBuyerMaker`. ([developers.binance.com](https://developers.binance.com/docs/binance-spot-api-docs/faqs/spot_glossary?utm_source=openai))  
- Binance developer community explanation: `isBuyerMaker=true` ⇒ sell-side initiated (seller taker), `false` ⇒ buy-side initiated. ([dev.binance.vision](https://dev.binance.vision/t/recent-trades-historical-and-aggregate/128?utm_source=openai))  
- Binance Academy: funding rates mechanism and periodic funding concept. ([academy.binance.com](https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets?utm_source=openai))  
- Borsa Istanbul Review (2025). *High-frequency dynamics of Bitcoin futures: An examination of market microstructure* (Binance perps; intraday aggregations; caution about 1m noise). ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S2214845025001188?utm_source=openai))  

### NautilusTrader backtesting and timestamp/execution semantics
- NautilusTrader docs: backtesting concepts, bar timestamp convention (`ts_init` close), OHLC simulation and slippage modeling. ([nautilustrader.io](https://nautilustrader.io/docs/nightly/concepts/backtesting/?utm_source=openai))  
- NautilusTrader docs: `ts_event` vs `ts_init` semantics and backtest ordering behavior. ([nautilustrader.io](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai))  
- NautilusTrader API reference: Backtest iterator tie-breaking via stream priority (`append_data`). ([nautilustrader.io](https://nautilustrader.io/docs/latest/api_reference/backtest/?utm_source=openai))

---
Learn more:
1. [How Often to Sample a Continuous-Time Process in the Presence of Market Microstructure Noise | NBER](https://www.nber.org/papers/w9611?utm_source=openai)
2. [Data | NautilusTrader Documentation](https://nautilustrader.io/docs/latest/concepts/data?utm_source=openai)
3. [VPIN and the Flash Crash | Kellogg School of Management](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/?utm_source=openai)
4. [What Are Funding Rates in Crypto Markets?](https://academy.binance.com/en/articles/what-are-funding-rates-in-crypto-markets?utm_source=openai)
5. [Designing realised kernels to measure the ex-post variation of equity prices in the presence of noise | Neil Shephard](https://shephard.scholars.harvard.edu/publications/designing-realised-kernels-measure-ex-post-variation-equity-prices-presence?utm_source=openai)
6. [Modeling and Forecasting Realized Volatility | NBER](https://www.nber.org/papers/w8160?utm_source=openai)
7. [Backtesting | NautilusTrader Documentation](https://nautilustrader.io/docs/latest/concepts/backtesting/?utm_source=openai)
8. [EconPapers: Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices](https://econpapers.repec.org/RePEc%3Aucp%3Ajnlbus%3Av%3A73%3Ay%3A2000%3Ai%3A3%3Ap%3A477-91?utm_source=openai)
9. [Backtest | NautilusTrader Documentation](https://nautilustrader.io/docs/latest/api_reference/backtest/?utm_source=openai)
10. [Efficient Estimation of Stochastic Volatility Using Noisy Observations: A Multi-Scale Approach](https://arxiv.org/abs/math/0411397?utm_source=openai)
11. [EconPapers: Microstructure Noise in the Continuous Case: The Pre-Averaging Approach - JLMPV-9](https://econpapers.repec.org/RePEc%3Aaah%3Acreate%3A2007-43?utm_source=openai)
12. [Power and Bipower Variation with Stochastic Volatility and Jumps | Journal of Financial Econometrics | Oxford Academic](https://academic.oup.com/jfec/article/2/1/1/960705?utm_source=openai)
13. [EconPapers: A Simple Approximate Long-Memory Model of Realized Volatility](https://econpapers.repec.org/RePEc%3Aoup%3Ajfinec%3Av%3A7%3Ay%3A2009%3Ai%3A2%3Ap%3A174-196?utm_source=openai)
14. [Realized GARCH: a joint model for returns and realized measures of volatility - Aarhus University](https://pure.au.dk/portal/en/publications/realized-garch-a-joint-model-for-returns-and-realized-measures-of?utm_source=openai)
15. [EconPapers: What can explain the price, volatility and trading volume of Bitcoin?](https://econpapers.repec.org/RePEc%3Aeee%3Afinlet%3Av%3A29%3Ay%3A2019%3Ai%3Ac%3Ap%3A255-265?utm_source=openai)
16. [Improving the realized GARCH’s volatility forecast for Bitcoin with jump-robust estimators - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1062940820300620?utm_source=openai)
17. [Bitcoin volatility predictability–The role of jumps and regimes - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1544612322000162?utm_source=openai)
18. [Price Impact of Order Book Events | Journal of Financial Econometrics | Oxford Academic](https://academic.oup.com/jfec/article/12/1/47/816163?utm_source=openai)
19. [Flow Toxicity and Liquidity in a High-frequency World | The Review of Financial Studies | Oxford Academic](https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929?utm_source=openai)
20. [High-frequency dynamics of Bitcoin futures: An examination of market microstructure - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214845025001188?utm_source=openai)
21. [Spot Glossary | Binance Open Platform](https://developers.binance.com/docs/binance-spot-api-docs/faqs/spot_glossary?utm_source=openai)
22. [Optimal execution of portfolio transactions - Journal of Risk](https://www.risk.net/journal-of-risk/technical-paper/2161150/optimal-execution-portfolio-transactions?utm_source=openai)
23. [A Reality Check for Data Snooping | The Econometric Society](https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping?utm_source=openai)
24. ["The Probability of Backtest Overfitting" by David H. Bailey, Jonathan Borwein et al.](https://scholarworks.wmich.edu/math_pubs/42/?utm_source=openai)
25. [The deflated Sharpe ratio : correcting for selection bias, backtest overfitting, and non-normality - EconBiz](https://www.econbiz.de/Record/the-deflated-sharpe-ratio-correcting-for-selection-bias-backtest-overfitting-and-non-normality-bailey-david/10011433463?utm_source=openai)
26. [Backtesting | NautilusTrader Documentation](https://nautilustrader.io/docs/nightly/concepts/backtesting/?utm_source=openai)
27. [Ultra high frequency volatility estimation with dependent microstructure noise](https://ideas.repec.org/a/eee/econom/v160y2011i1p160-175.html?utm_source=openai)
28. [EconPapers: VPIN and the Flash Crash: A rejoinder](https://econpapers.repec.org/RePEc%3Aeee%3Afinmar%3Av%3A17%3Ay%3A2014%3Ai%3Ac%3Ap%3A47-52?utm_source=openai)
29. [Continuous Auctions and Insider Trading | The Econometric Society](https://www.econometricsociety.org/publications/econometrica/browse/1985/11/01/continuous-auctions-and-insider-trading?utm_source=openai)
30. [EconPapers: Optimal trading strategy and supply/demand dynamics](https://econpapers.repec.org/RePEc%3Aeee%3Afinmar%3Av%3A16%3Ay%3A2013%3Ai%3A1%3Ap%3A1-32?utm_source=openai)
31. [. . . and the Cross-Section of Expected Returns | NBER](https://www.nber.org/papers/w20592?utm_source=openai)
32. [recent trades, historical and aggregate - Spot/Margin API - Binance Developer Community](https://dev.binance.vision/t/recent-trades-historical-and-aggregate/128?utm_source=openai)
