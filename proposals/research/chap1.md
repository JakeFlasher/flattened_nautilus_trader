# High-Frequency Volatility Arbitrage in BTCUSDT Perpetual Futures  
## Technical Whitepaper + System Architecture Specification (HFT / Microstructure)  
**Author:** Principal Quantitative Scientist (PhD), Proprietary Trading Research

---

## Table of Contents (50-page technical specification blueprint)

> **Note:** Page counts are indicative (assuming ~500–700 words/page plus equations/figures).

### Front Matter (pp. 1–6)
0. **Title, abstract, executive thesis** (p. 1)  
1. **Notation, units, and data contracts** (pp. 2–4)  
   1.1. Time conventions (UTC, ms/ns), event-time vs. clock-time  
   1.2. Price types (trade, mid, mark) and microstructure conventions  
   1.3. Returns conventions and log transforms  
2. **Threat model: what breaks in crypto microstructure** (pp. 5–6)  
   2.1. Bid–ask bounce, discreteness, latency, partial fills  
   2.2. 24/7 trading: “no open/close” implications  
   2.3. Regime switching: funding, liquidation cascades

---

### Chapter 1 — Mathematical definition of alpha (pp. 7–22)
1. **Price formation model (latent efficient price + microstructure observation)** (pp. 7–10)  
   1.1. Latent price as an Itô semimartingale with jumps  
   1.2. Observed trade price contaminated by microstructure noise  
   1.3. Quadratic variation and what “realized volatility” estimates  
2. **Factor 1: Realized volatility (OHLC) for crypto perpetuals** (pp. 11–16)  
   2.1. Rogers–Satchell variance: drift-independence lemma  
   2.2. Yang–Zhang volatility: definition, drift-independence, “overnight” adaptation to 24/7  
   2.3. Microstructure noise vs. sampling frequency (why 1m is not “HF” in theory but is practical)  
3. **Factor 2: Flow toxicity (OFI + VPIN) from `aggTrades` only** (pp. 17–20)  
   3.1. Trade-sign inference from `is_buyer_maker`  
   3.2. OFI estimator in trade-time and time-bars  
   3.3. VPIN via volume clock + bulk volume classification  
   3.4. Known failure modes / critiques (mechanical correlation with volume)  
4. **Signal fusion: latent alpha state and target position** (pp. 21–22)  
   4.1. Multivariate Kalman filter for online fusion  
   4.2. Converting posterior alpha into a target position signal $\theta_t$

---

# Chapter 1 — Mathematical Definition of Alpha

## 1.1 Notation and the time scales that matter

We work with three time scales:

1. **Trade-time / event-time** indexed by $i \in \{1,2,\dots\}$, where each event corresponds to one row in:
   - `data/raw/futures/daily/aggTrades/BTCUSDT-aggTrades-YYYY-MM-DD.csv`

2. **Bar-time** indexed by $k \in \{1,2,\dots\}$, where each bar corresponds to one row in:
   - `data/raw/futures/daily/klines_1m/BTCUSDT-1m-YYYY-MM-DD.csv`

3. **Decision-time** $t$ at which we update our target position $\theta_t$. In a practical HFT engine, $t$ may coincide with every bookTicker update, every trade, or a fixed-rate timer; mathematically, we treat $t$ as continuous but implement in discrete ticks.

### Basic objects

- $P_t$: **latent efficient price** (unobserved) at time $t$.  
- $Y_t$: **observed price** (trade price, midquote, etc.) at time $t$.  
- $X_t = \log P_t$, $Z_t = \log Y_t$.

For bars (1-minute):

- $O_k, H_k, L_k, C_k$: open/high/low/close prices in bar $k$.
- Bar return (close-to-close) in log space:
  $$ r_k := \log\!\left(\frac{C_k}{C_{k-1}}\right). $$

For aggTrades:

- Each record $i$ has:
  - price $p_i$,
  - quantity $q_i$,
  - timestamp $\tau_i$ (ms),
  - boolean `is_buyer_maker` (we denote $\mathrm{BM}_i \in \{0,1\}$).

We will infer a **trade sign** $s_i \in \{+1,-1\}$ using `is_buyer_maker` (details §1.3).

---

## 1.2 Theoretical framework: latent price + microstructure observation

### Definition 1 (Latent efficient log-price process)

Model the latent efficient log-price $X_t$ as an Itô semimartingale with jumps:

$$
dX_t = \mu_t\,dt + \sigma_t\,dW_t + dJ_t,
$$

where:

- $\mu_t$ is the (possibly stochastic) drift,
- $\sigma_t > 0$ is the instantaneous volatility,
- $W_t$ is standard Brownian motion,
- $J_t$ is a pure-jump process (to capture liquidation cascades / news jumps typical in crypto).

This is the minimum mathematically disciplined model that admits:
- diffusion-driven micro-moves,
- stochastic volatility,
- jump risk.

### Definition 2 (Observed price under microstructure noise)

Observed prices are contaminated by microstructure noise:

$$
Z_t = X_t + \varepsilon_t,
$$

where $\varepsilon_t$ captures bid–ask bounce, discreteness, latency, and trade price formation frictions.

This decomposition is the standard starting point for “microstructure noise” analysis. A central implication is that **sampling more frequently is not always better** when one estimates volatility from observed returns: realized variance can be biased by $\varepsilon_t$. ([nber.org](https://www.nber.org/papers/w9611?utm_source=openai))

### Lemma 1 (Quadratic variation and integrated variance)

Define the **integrated variance** (continuous component) over horizon \([t,t+T]\):

$$
\mathrm{IV}(t,t+T) := \int_t^{t+T} \sigma_s^2\,ds.
$$

If $X_t$ is continuous (ignore jumps for a moment) and we observe $X_{t_j}$ on a refining grid with mesh $\max_j |t_{j+1}-t_j| \to 0$, then:

$$
\sum_{j} \left(X_{t_{j+1}} - X_{t_j}\right)^2 \;\xrightarrow{\;p\;}\; \mathrm{IV}(t,t+T).
$$

**Proof logic (sketch):** This is the defining property of quadratic variation for continuous semimartingales; realized variance converges to quadratic variation, which equals the integrated variance for the Brownian component. This underpins realized volatility modeling and forecasting. ([nber.org](https://www.nber.org/papers/w8160?utm_source=openai))

### Lemma 2 (Microstructure noise makes “sample as fast as possible” pathological)

If instead we observe $Z_t = X_t + \varepsilon_t$ with i.i.d.-like noise, then observed returns satisfy:

$$
\Delta Z_{t_j} = \Delta X_{t_j} + \Delta \varepsilon_{t_j},
$$

and the realized variance from $Z$ includes terms like:

$$
\sum_j (\Delta \varepsilon_{t_j})^2,
$$

which can **dominate** as the grid refines. Thus, “max frequency” can inflate volatility estimates unless noise is modeled/filtered. ([nber.org](https://www.nber.org/papers/w9611?utm_source=openai))

**Crypto-specific remark:** Empirical work on Bitcoin futures microstructure explicitly cautions that very high sampling (even 1-minute) can be distorted by market microstructure effects (bid–ask bounce, discreteness, order book frictions), and that coarser intervals (e.g., 5-minute) often behave more stably in realized-volatility microstructure studies. ([sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S2214845025001188?utm_source=openai))

---

## 1.3 Factor 1 — Realized volatility (OHLC): Rogers–Satchell and Yang–Zhang

We must estimate short-horizon variance robustly **without** full L2/L3 book reconstruction and without tick-by-tick midquotes. Your dataset provides 1-minute OHLC bars, which is a pragmatic compromise: it aggregates away much of the bid–ask bounce while preserving intraminute range information.

### 1.3.1 Rogers–Satchell (RS): drift independence from OHLC

#### Definition 3 (Log OHLC increments within a bar)

For a bar $k$, define:

$$
u_k := \log\!\left(\frac{H_k}{O_k}\right), \quad
d_k := \log\!\left(\frac{L_k}{O_k}\right), \quad
c_k := \log\!\left(\frac{C_k}{O_k}\right).
$$

Note: $u_k \ge 0$, $d_k \le 0$, $c_k \in \mathbb{R}$.

#### Definition 4 (Rogers–Satchell per-bar variance contribution)

The Rogers–Satchell variance contribution for bar $k$ is:

$$
\widehat{\sigma}^2_{\mathrm{RS},k}
:=
u_k(u_k - c_k) + d_k(d_k - c_k).
$$

Over a window of $n$ bars (e.g., last $n$ minutes), the RS estimator is:

$$
\widehat{\sigma}^2_{\mathrm{RS}}(t)
:=
\frac{1}{n}\sum_{k=t-n+1}^{t} \widehat{\sigma}^2_{\mathrm{RS},k}.
$$

This is the canonical OHLC RS form attributable to Rogers–Satchell’s drift-robust variance estimation principle. ([researchgate.net](https://www.researchgate.net/publication/38362991_Estimating_Variance_From_High_Low_and_Closing_Prices?utm_source=openai))

#### Lemma 3 (RS is unbiased w.r.t. drift in GBM limit)

Assume within a bar the log-price follows:

$$
X_s = X_0 + \mu s + \sigma W_s, \quad s\in[0,\Delta],
$$

with constant drift $\mu$ and volatility $\sigma$ over the bar. Then the RS estimator satisfies:

$$
\mathbb{E}\!\left[\widehat{\sigma}^2_{\mathrm{RS},k}\right] = \sigma^2 \Delta,
$$

i.e., it does not depend on $\mu$.

**Proof logic (sketch):** The RS functional is constructed to cancel drift terms when integrating over the joint law of \((H,L,C)\) under Brownian motion with drift; it uses mixed products of high/open/close and low/open/close log-ratios that eliminate the drift contribution. This is precisely why RS is used when $\mu \neq 0$ (trending markets). ([researchgate.net](https://www.researchgate.net/publication/38362991_Estimating_Variance_From_High_Low_and_Closing_Prices?utm_source=openai))

> **Microstructure note:** RS (like most OHLC estimators) assumes “true” high/low are observed. With discretely sampled markets and aggregation, we observe maxima/minima over sampled trades, not the continuous path extrema. This induces error; however range-based estimators can remain comparatively robust in the presence of microstructure noise (in simulation/empirical studies). ([econpapers.repec.org](https://econpapers.repec.org/RePEc%3Awly%3Ajfutmk%3Av%3A26%3Ay%3A2006%3Ai%3A3%3Ap%3A297-313?utm_source=openai))

---

### 1.3.2 Yang–Zhang (YZ): combining “overnight”, close-to-open, and RS

The Yang–Zhang framework was designed to address two problems simultaneously:

1. **Drift bias** (like RS fixes),
2. **Opening jumps / session gaps** (RS alone does not).

This is classically formulated for markets with a daily open/close. Crypto is 24/7, but we can still define *windows* (e.g., UTC day) and treat the boundary as a “pseudo-open” to capture discontinuities due to:
- liquidity regime changes across global sessions,
- funding timestamps / liquidation clusters,
- scheduled macro announcements (which empirically cluster in crypto too). ([arxiv.org](https://arxiv.org/abs/2306.17095?utm_source=openai))

#### Definition 5 (Windowing for 24/7 markets)

Let a *session* be a fixed calendar partition, e.g.:

- Session $d$ is \([d\text{ 00:00:00 UTC}, (d+1)\text{ 00:00:00 UTC})\).

Define session open $O_d$ as the first 1m bar open in session $d$, and session close $C_d$ as the last 1m bar close in session $d$.

This is not “exchange open/close” (crypto has none); it is a **statistical window boundary**.

#### Definition 6 (Close-to-open and open-to-close log returns)

For session $d$:

- “Overnight” (boundary) return:
  $$
  r^{(o)}_d := \log\!\left(\frac{O_d}{C_{d-1}}\right).
  $$

- “Open-to-close” return:
  $$
  r^{(c)}_d := \log\!\left(\frac{C_d}{O_d}\right).
  $$

Within-session, define Rogers–Satchell variance $\widehat{\sigma}^2_{\mathrm{RS},d}$ computed from intraday OHLC bars (here, 1m bars aggregated over that session).

#### Definition 7 (Yang–Zhang volatility estimator)

Yang–Zhang proposes (for a window of $m$ sessions) an estimator of total variance:

$$
\widehat{\sigma}^2_{\mathrm{YZ}}
=
\widehat{\sigma}^2_{o}
+
k\,\widehat{\sigma}^2_{c}
+
(1-k)\,\widehat{\sigma}^2_{\mathrm{RS}},
$$

where:

- $\widehat{\sigma}^2_o$ is the sample variance of $r^{(o)}_d$ over the window,
- $\widehat{\sigma}^2_c$ is the sample variance of $r^{(c)}_d$ over the window,
- $\widehat{\sigma}^2_{\mathrm{RS}}$ is the average RS variance over the window,
- $k$ is a weighting factor derived in the Yang–Zhang paper (chosen to minimize variance among a class of estimators with the stated invariances). ([econpapers.repec.org](https://econpapers.repec.org/article/ucpjnlbus/v_3a73_3ay_3a2000_3ai_3a3_3ap_3a477-91.htm?utm_source=openai))

> The defining property is **drift independence** and **consistency with opening jumps** in the continuous limit. ([econpapers.repec.org](https://econpapers.repec.org/article/ucpjnlbus/v_3a73_3ay_3a2000_3ai_3a3_3ap_3a477-91.htm?utm_source=openai))

#### Lemma 4 (Why YZ dominates close-to-close under microstructure + regime boundary effects)

Let $\widehat{\sigma}^2_{\mathrm{CC}}$ denote close-to-close variance from $\log(C_d/C_{d-1})$. Then, informally:

- $\widehat{\sigma}^2_{\mathrm{CC}}$ uses **one point per session**.
- $\widehat{\sigma}^2_{\mathrm{YZ}}$ uses:
  - boundary discontinuity proxy $r^{(o)}_d$,
  - within-session directional move $r^{(c)}_d$,
  - within-session extremes via RS.

Thus $\widehat{\sigma}^2_{\mathrm{YZ}}$ is strictly richer (lower estimator variance under the model class it targets), and is designed to be drift independent and robust to boundary jumps. ([econpapers.repec.org](https://econpapers.repec.org/article/ucpjnlbus/v_3a73_3ay_3a2000_3ai_3a3_3ap_3a477-91.htm?utm_source=openai))

**Crypto nuance (important):** “Overnight” is not literal; it is “across the boundary of our partition.” In a 24/7 market, the boundary return can still capture predictable discontinuities driven by global session liquidity and macro release schedules. ([arxiv.org](https://arxiv.org/abs/2306.17095?utm_source=openai))

---

### 1.3.3 Deliverable: factor construction from your *exact* files

For every UTC day $d$:

1. Load:
   - `data/raw/futures/daily/klines_1m/BTCUSDT-1m-YYYY-MM-DD.csv`

2. For every 1-minute bar $k$ in that day:
   - compute $u_k, d_k, c_k$ and $\widehat{\sigma}^2_{\mathrm{RS},k}$.

3. Aggregate RS intraday variance:
   $$
   \widehat{\sigma}^2_{\mathrm{RS},d} := \sum_{k \in d} \widehat{\sigma}^2_{\mathrm{RS},k}.
   $$

4. Construct session open/close $O_d, C_d$ from first/last bar of day and compute:
   - $r^{(o)}_d$, $r^{(c)}_d$.

5. Over a rolling window of $m$ days, compute:
   - $\widehat{\sigma}^2_o, \widehat{\sigma}^2_c, \widehat{\sigma}^2_{\mathrm{RS}}$,
   - then $\widehat{\sigma}^2_{\mathrm{YZ}}$.

Define **Factor 1** as (one of):
- variance factor: $F^{(\mathrm{vol})}_t := \widehat{\sigma}^2_{\mathrm{YZ}}(t)$,
- volatility factor: $\sqrt{\widehat{\sigma}^2_{\mathrm{YZ}}(t)}$,
- standardized surprise:
  $$
  Z^{(\mathrm{vol})}_t := \frac{\widehat{\sigma}^2_{\mathrm{YZ}}(t) - \mathrm{EMA}(\widehat{\sigma}^2_{\mathrm{YZ}})(t)}{\mathrm{MAD}(\widehat{\sigma}^2_{\mathrm{YZ}})(t)}.
  $$

The final choice depends on how you fuse with flow toxicity; we will do this in §1.5.

---

## 1.4 Factor 2 — Flow toxicity from `aggTrades`: OFI and VPIN

### 1.4.1 Trade sign from `is_buyer_maker` (Binance convention)

Your `aggTrades` schema includes `is_buyer_maker`.

Empirically, in Binance-style data, `isBuyerMaker = True` means the **buyer provided liquidity (maker)**, so the **seller was the taker**, i.e. the trade was *seller-initiated*. Conversely, `isBuyerMaker = False` corresponds to a *buyer-initiated* (taker buy) trade.

We therefore define trade sign:

$$
s_i :=
\begin{cases}
+1, & \mathrm{BM}_i = 0 \quad (\text{buyer-initiated / taker buy}), \\
-1, & \mathrm{BM}_i = 1 \quad (\text{seller-initiated / taker sell}). \\
\end{cases}
$$

This is the simplest **self-contained** sign inference possible under your constraint “`aggTrades` only”.

---

### 1.4.2 Order Flow Imbalance (OFI) from `aggTrades`

The classical OFI in Cont–Kukanov–Stoikov is defined from L1 queue changes at best bid/ask, not from trades alone. ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))

But you explicitly constrain us to aggTrades for the “flow” feature class. Therefore we define a **trade-flow imbalance** (call it $\mathrm{tOFI}$):

#### Definition 8 (Trade-flow OFI over a time interval)

Let $[t-\Delta, t)$ be a fixed clock-time bucket (e.g., \(\Delta=1\) second). Define:

- Buyer-initiated volume:
  $$
  V^+_{t,\Delta} := \sum_{i:\,\tau_i \in [t-\Delta,t)} \mathbf{1}\{s_i=+1\}\,q_i,
  $$
- Seller-initiated volume:
  $$
  V^-_{t,\Delta} := \sum_{i:\,\tau_i \in [t-\Delta,t)} \mathbf{1}\{s_i=-1\}\,q_i.
  $$

Then:

$$
\mathrm{tOFI}_{t,\Delta} := V^+_{t,\Delta} - V^-_{t,\Delta}.
$$

A normalized version (bounded in \([-1,1]\)):

$$
\widetilde{\mathrm{tOFI}}_{t,\Delta}
:=
\frac{V^+_{t,\Delta} - V^-_{t,\Delta}}{V^+_{t,\Delta} + V^-_{t,\Delta} + \epsilon}.
$$

#### Lemma 5 (Why OFI is a first-principles microstructure predictor)

A core empirical microstructure fact is that **short-horizon price changes are driven by order flow imbalance**, and in particular, OFI is often more robust than raw volume in explaining price moves. ([arxiv.org](https://arxiv.org/abs/1011.6402?utm_source=openai))

**Proof logic (adaptation):** In a linearized limit order book, the best bid/ask queues are depleted by aggressive flow and replenished by passive flow; net imbalance at the top of book shifts the midprice when one side is exhausted. Trades are the *executed* component of aggressive flow; thus signed trade imbalance is a noisy but direct proxy for the same supply/demand asymmetry that OFI formalizes in full order-book event space.

---

### 1.4.3 VPIN: volume-synchronized probability of informed trading (trade-only implementation)

VPIN was designed explicitly for high-frequency contexts by operating in **volume time** rather than clock time, and by estimating toxicity from volume imbalance without requiring unobservable intermediate parameters. ([academic.oup.com](https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929?utm_source=openai))

#### Definition 9 (Volume clock and buckets)

Fix a **bucket volume** $V_b > 0$ (units: BTC contracts or BTC quantity). Construct buckets sequentially by consuming trades in time order:

Let $B_j$ be bucket $j$ consisting of consecutive trades whose quantities sum to $V_b$ (except possibly the last trade is fractionally split).

Formally, let $\mathcal{I}_j$ be the (possibly fractional) set of trade indices allocated to bucket $j$, such that:

$$
\sum_{i \in \mathcal{I}_j} q_i = V_b.
$$

Within bucket $j$, define:

$$
V^{+}_j := \sum_{i \in \mathcal{I}_j} \mathbf{1}\{s_i=+1\}\,q_i, \quad
V^{-}_j := \sum_{i \in \mathcal{I}_j} \mathbf{1}\{s_i=-1\}\,q_i,
$$

so $V^{+}_j + V^{-}_j = V_b$.

#### Definition 10 (Bucket imbalance)

Define bucket imbalance:

$$
I_j := |V^{+}_j - V^{-}_j|.
$$

#### Definition 11 (VPIN)

Let $n$ be the rolling window length in number of buckets. VPIN at bucket time $j$ is:

$$
\mathrm{VPIN}_j := \frac{1}{n V_b}\sum_{k=j-n+1}^{j} I_k.
$$

So $\mathrm{VPIN}_j \in [0,1]$.

This is the “pure” VPIN construction: rolling average of normalized volume imbalance in equal-volume buckets. ([academic.oup.com](https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929?utm_source=openai))

---

### 1.4.4 Bulk volume classification under your schema constraint

Easley–López de Prado–O’Hara discuss a “bulk volume classification” procedure to classify trades as buys/sells in high-frequency markets. ([academic.oup.com](https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929?utm_source=openai))

Your dataset already gives `is_buyer_maker`, which is a direct trade-direction proxy. Therefore:

- We do **not** need Lee–Ready-style inference (quote rule) or tick rule.
- We do **not** need bulk classification based on returns distribution.

We can interpret our sign $s_i$ as the output of the classification procedure and proceed.

This is a material simplification (and a potential source of bias if `is_buyer_maker` behaves differently than assumed), but it is the only disciplined approach under your “no external APIs” + “aggTrades only” constraints.

---

### 1.4.5 VPIN caveats (must be included in a serious spec)

A rigorous spec must note that VPIN has well-known critiques: its predictive power can be dominated by mechanical relationships with volume/trading intensity, and it may not behave as claimed around stress events depending on implementation details. ([kellogg.northwestern.edu](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/?utm_source=openai))

**Practical implication for our system design:**  
We treat VPIN not as “truth” but as a **risk gate** and **execution control input** (e.g., widen spreads / accelerate liquidation avoidance), and we validate its incremental predictive power out-of-sample relative to simpler signed-flow metrics (e.g., \(\widetilde{\mathrm{tOFI}}\), signed volume z-scores).

---

## 1.5 Signal combination into a single target position $\theta_t$

You required either:

- a multivariate Kalman filter **or**
- a z-score ensemble.

We will define a **state-space model** and a **Kalman filter** because it provides:

- online updates,
- uncertainty quantification,
- principled shrinkage under noise (critical in HFT crypto microstructure where features are noisy).

### 1.5.1 What is the latent state?

We need a latent object that maps naturally to a position. The minimal choice is:

- $\alpha_t$: *instantaneous expected return (drift) over the next decision interval*, measured in log-return units per unit time.

This is not claiming markets have stable drift; it is a local predictive component induced by microstructure (order flow imbalance) and transient volatility regimes.

### Definition 12 (Alpha state dynamics)

Assume $\alpha_t$ follows a mean-reverting AR(1) in discrete time at decision timestamps $t_k$:

$$
\alpha_{k+1} = \phi \alpha_k + \eta_k,
\quad |\phi|<1,
\quad \eta_k \sim \mathcal{N}(0, Q).
$$

Interpretation:

- $\phi$ encodes **alpha half-life** (short for HFT),
- $Q$ is innovation variance capturing regime shifts.

### Definition 13 (Observation vector from your two factors)

At time $t_k$, define feature vector $\mathbf{f}_k \in \mathbb{R}^2$:

1. Volatility surprise (from Yang–Zhang):
   $$
   f^{(1)}_k := Z^{(\mathrm{vol})}_{t_k}.
   $$

2. Toxicity-adjusted signed flow:
   We need a scalar that is large when (a) there is directional imbalance and (b) toxicity is high.
   Define:
   $$
   f^{(2)}_k := \widetilde{\mathrm{tOFI}}_{t_k,\Delta}\cdot g(\mathrm{VPIN}_{t_k}),
   $$
   where a natural choice is a monotone map such as:
   $$
   g(v) := \max\{0, v - v_0\},
   $$
   with $v_0$ a “toxicity threshold” (e.g., a high quantile of VPIN).

### Definition 14 (Linear observation model)

Let observations be:

$$
\mathbf{f}_k = \mathbf{H}\alpha_k + \mathbf{v}_k,
\quad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}).
$$

Here $\mathbf{H} \in \mathbb{R}^{2 \times 1}$ is a loading vector (two scalars), and $\mathbf{R}$ is the observation noise covariance.

> This model says: features are noisy linear sensors of a latent drift-like alpha.

### Kalman filter recursion (for completeness)

Let prior estimate at step $k$ be $\hat{\alpha}_{k|k-1}$ with variance $P_{k|k-1}$.

**Predict:**
$$
\hat{\alpha}_{k|k-1} = \phi \hat{\alpha}_{k-1|k-1}, \quad
P_{k|k-1} = \phi^2 P_{k-1|k-1} + Q.
$$

**Update:**
Innovation:
$$
\mathbf{y}_k = \mathbf{f}_k - \mathbf{H}\hat{\alpha}_{k|k-1}.
$$

Innovation covariance:
$$
\mathbf{S}_k = \mathbf{H}P_{k|k-1}\mathbf{H}^\top + \mathbf{R}.
$$

Kalman gain:
$$
\mathbf{K}_k = P_{k|k-1}\mathbf{H}^\top \mathbf{S}_k^{-1}.
$$

Posterior:
$$
\hat{\alpha}_{k|k} = \hat{\alpha}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k,
$$
$$
P_{k|k} = (1 - \mathbf{K}_k \mathbf{H}) P_{k|k-1}.
$$

---

### 1.5.2 From posterior alpha to target position $\theta_t$

We now define $\theta_k$ (position in contracts / BTC notional units) as a **risk-scaled** function of posterior alpha and posterior variance.

A minimal “Kelly-like” local mapping (without yet doing full Chapter 3) is:

$$
\theta_k
=
\mathrm{clip}\!\left(
\frac{\hat{\alpha}_{k|k}}{\gamma \, \widehat{\sigma}^2_{\mathrm{YZ}}(t_k)},
\;-\theta_{\max},\;+\theta_{\max}
\right),
$$

where:

- $\widehat{\sigma}^2_{\mathrm{YZ}}(t_k)$ is Factor 1,
- $\gamma>0$ is a risk-aversion / scaling constant,
- $\theta_{\max}$ is a hard cap.

**Interpretation:**  
- High predicted drift → larger position.  
- High realized variance → smaller position (volatility targeting / risk-normalization).  
- This is the minimal principled bridge from “alpha estimate” to “position.”

---

## 1.6 Summary of Chapter 1 deliverables (formal)

### Deliverable A — Factor 1 (Realized volatility)

From `data/raw/futures/daily/klines_1m/BTCUSDT-1m-*.csv`:

- Compute Rogers–Satchell intrabar variance contributions $\widehat{\sigma}^2_{\mathrm{RS},k}$.
- Aggregate into Yang–Zhang variance $\widehat{\sigma}^2_{\mathrm{YZ}}(t)$ over rolling day windows. ([econpapers.repec.org](https://econpapers.repec.org/article/ucpjnlbus/v_3a73_3ay_3a2000_3ai_3a3_3ap_3a477-91.htm?utm_source=openai))

### Deliverable B — Factor 2 (Flow toxicity)

From `data/raw/futures/daily/aggTrades/BTCUSDT-aggTrades-*.csv`:

- Infer trade signs $s_i$ from `is_buyer_maker`.
- Compute trade-flow imbalance $\widetilde{\mathrm{tOFI}}_{t,\Delta}$.
- Compute VPIN in volume time using equal-volume buckets $V_b$. ([academic.oup.com](https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929?utm_source=openai))

### Deliverable C — Single target position signal

- Fuse features into latent $\alpha_t$ via Kalman filtering.
- Map $\hat{\alpha}_{t|t}$ and $\widehat{\sigma}^2_{\mathrm{YZ}}(t)$ into $\theta_t$ via risk-scaled clipping.

---

If you want, I will proceed next with **Chapter 2 (NautilusTrader Architecture)** and give (a) the exact `CSVDataLoader` logic for your `data/raw/futures/*` directory, (b) a strict event-driven update design (`on_bar` vs `on_quote_tick`), and (c) the TWAP-with-alpha-decay execution algorithm tied directly to $\theta_t$ and VPIN-driven toxicity gating (citing the Nautilus docs you provided for Actors, Cache, Execution flow, and MessageBus behavior).

---
Learn more:
1. [How Often to Sample a Continuous-Time Process in the Presence of Market Microstructure Noise | NBER](https://www.nber.org/papers/w9611?utm_source=openai)
2. [Modeling and Forecasting Realized Volatility | NBER](https://www.nber.org/papers/w8160?utm_source=openai)
3. [High-frequency dynamics of Bitcoin futures: An examination of market microstructure - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214845025001188?utm_source=openai)
4. [(PDF) Estimating Variance From High, Low and Closing Prices](https://www.researchgate.net/publication/38362991_Estimating_Variance_From_High_Low_and_Closing_Prices?utm_source=openai)
5. [EconPapers: Testing range estimators of historical volatility](https://econpapers.repec.org/RePEc%3Awly%3Ajfutmk%3Av%3A26%3Ay%3A2006%3Ai%3A3%3Ap%3A297-313?utm_source=openai)
6. [Decomposing cryptocurrency high-frequency price dynamics into recurring and noisy components](https://arxiv.org/abs/2306.17095?utm_source=openai)
7. [EconPapers: Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices](https://econpapers.repec.org/article/ucpjnlbus/v_3a73_3ay_3a2000_3ai_3a3_3ap_3a477-91.htm?utm_source=openai)
8. [The Price Impact of Order Book Events](https://arxiv.org/abs/1011.6402?utm_source=openai)
9. [Flow Toxicity and Liquidity in a High-frequency World | The Review of Financial Studies | Oxford Academic](https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929?utm_source=openai)
10. [VPIN and the Flash Crash | Kellogg School of Management](https://www.kellogg.northwestern.edu/faculty/research/detail/2014/vpin-and-the-flash-crash/?utm_source=openai)
