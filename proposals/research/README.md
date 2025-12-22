# Liquidity Pressure Index

## Ranking Factors
spread/open_interest/traded_volume/bid_ask_volume > inflow/outflow > funding_rate


## Can Rogers–Satchell variance be used for estimating crypto asset's liquidity pressure index?
Yes—but only **as an input (volatility component)**, not as a **stand‑alone** “liquidity pressure index.”

Rogers–Satchell (RS) is a **range-based volatility estimator** built from **OHLC**. Liquidity pressure, in most definitions, is about **how much price moves per unit of liquidity supplied/consumed** (depth, volume, spread, order‑flow imbalance, price impact). Volatility and liquidity are related, but **volatility ≠ liquidity**: high volatility can come from information/jumps even in very liquid markets, and illiquidity can exist with low volatility in quiet periods.

Below is how RS *can* be used sensibly for a crypto “liquidity pressure” proxy.

---

## 1) What RS variance measures (and why it’s relevant)

For a bar \(t\) with open \(O_t\), high \(H_t\), low \(L_t\), close \(C_t\) (use log prices), the Rogers–Satchell per‑bar variance estimator is commonly written as:

\[
v^{RS}_t
=
\ln\!\left(\frac{H_t}{O_t}\right)\ln\!\left(\frac{H_t}{C_t}\right)
+
\ln\!\left(\frac{L_t}{O_t}\right)\ln\!\left(\frac{L_t}{C_t}\right)
\]

Intuition:

- It uses **high/low** (more information than close‑to‑close).
- It is designed to be **robust to drift** (helpful when there’s a trend inside the bar).
- In crypto, the **high–low range often widens** when the book is thin or when there are “wicks” caused by liquidity gaps—so RS can pick up liquidity stress.

But RS alone is still “how much price varied,” not “how hard it was to trade.”

---

## 2) Turning RS into a liquidity-pressure proxy (OHLCV-only)

If you only have **OHLCV** (no order book), the most defensible way is:

> **Liquidity pressure ≈ volatility per unit traded value (or per unit volume).**

This is the same idea as **Amihud illiquidity** \(|r|/\$Vol\), but using a **better volatility estimate** than \(|r|\).

### A practical RS-based LPI definition
Pick a bar size (e.g., 1m, 5m, 15m). For each bar \(t\):

- Compute \(v^{RS}_t\) from OHLC.
- Compute dollar volume:  
  \[
  DV_t = \text{Volume}_t \times \text{VWAP}_t
  \]
  If VWAP is unavailable, use \(C_t\) or \((H_t+L_t+C_t)/3\).

Then define an index like:

\[
LPI_t = \frac{\sqrt{v^{RS}_t}}{DV_t}
\qquad\text{or}\qquad
LPI_t = \frac{v^{RS}_t}{DV_t}
\]

Notes:

- \(\sqrt{v^{RS}_t}\) gives “volatility” units; \(v^{RS}_t\) gives “variance” units. Either can work—choose what behaves better empirically for you.
- Often you’ll rescale for readability, e.g. \(LPI_t \times 10^6\).

**Interpretation:** when **price range/variance is high** relative to **traded value**, it suggests the market is moving a lot “per dollar traded,” consistent with liquidity pressure / high price impact.

### Smoothing (recommended)
Liquidity measures are noisy. Use a rolling mean/median:

\[
\overline{LPI}_{t}^{(k)} = \text{median}\left(LPI_{t-k+1},\dots,LPI_t\right)
\]

Median is often more robust in crypto due to spikes.

---

## 3) If you *do* have order book or trades, RS is still only one piece

If your “liquidity pressure index” is meant to reflect **actual trading conditions** (slippage, depth depletion, spread widening), you usually want order-book or trade-level inputs:

- **Effective spread / quoted spread**
- **Depth at best bid/ask** (or within X bps)
- **Order book imbalance**
- **Price impact** coefficients (Kyle’s \(\lambda\), etc.)
- **Realized volatility** from trades/midprice changes (microstructure-aware)

In that world, RS can be used as a **volatility normalization** term, e.g.:

\[
LPI_t
=
\frac{\text{(spread or impact or imbalance)}}{\text{expected volatility}}
\quad\text{or}\quad
\text{impact} \times \sqrt{v^{RS}_t}
\]

…but RS should not replace depth/spread/impact measures.

---

## 4) Crypto-specific gotchas (important)

### (a) OHLC “wicks” can be data artifacts
RS is sensitive to high/low extremes. In crypto, a single bad print, exchange glitch, or thin book can create a massive wick.

Mitigations:
- Filter trades used to form OHLC (exclude out-of-sequence / obvious outliers).
- Winsorize the log ranges \(\ln(H/O)\), \(\ln(O/L)\) per asset.
- Use **midprice OHLC** (from top-of-book) instead of last-trade OHLC if you have it.

### (b) 24/7 trading makes “daily open” arbitrary
RS works fine on intraday bars, but for daily bars, the “open” depends on where you cut the day (UTC, exchange local time). Pick one convention (UTC is common) and stick with it.

### (c) RS assumes a diffusion-ish price process; crypto has jumps
Jumps can reflect information *or* liquidity gaps. If your goal is *liquidity pressure*, jumps are not necessarily “bad,” but they can blur interpretation. Consider also tracking:
- jump indicators (e.g., range vs close-to-close)
- separate “continuous” vs “jump” components if you’re doing research-grade work

---

## 5) A concrete “best practice” recipe (OHLCV only)

1. Choose venue & bar size: e.g., Binance BTC/USDT 1‑minute.
2. Compute per-bar \(v^{RS}_t\).
3. Compute dollar volume \(DV_t\).
4. Define \(LPI_t = \sqrt{v^{RS}_t} / DV_t\).
5. Smooth with rolling median/EMA (e.g., 30–120 bars).
6. Validate against something that *is* liquidity:
   - correlation with bid–ask spread (if you can get it later)
   - correlation with depth drops / slippage during stress events
   - whether it spikes during known liquidity shocks (exchange outages, liquidation cascades)

If it doesn’t line up with spreads/slippage/depth, then it’s mostly a volatility indicator, not a liquidity indicator.

---

## 6) What I need from you to be precise

When you say “liquidity pressure index,” do you mean something closer to:
 
1) **Amihud-style illiquidity** (price move per dollar volume),  
2) **order-book pressure** (imbalance/depth/spread), or  
3) a **custom risk index** used for monitoring?

And what data do you have: **OHLCV only**, **trades**, or **L2 order book** (and at what frequency)?

If you tell me your intended definition + available data, I can propose a specific RS-based formulation (and alternatives like Parkinson/Garman–Klass/realized volatility) that matches your goal and won’t be accidentally measuring the wrong thing.

## Can Yang–Zhang volatility estimator be used to estimate liquidity pressure index in crypto market?
Yang–Zhang (YZ) is fundamentally a **price-volatility estimator**, not a direct liquidity measure. So:

- **As a standalone “liquidity pressure index” (LPI):** *Not really.* It measures **variance of returns** using OHLC information; liquidity pressure is about **market depth, spreads, and price impact of order flow**.
- **As an input / component inside an LPI (especially if you only have OHLCV):** *Yes, it can be useful*, because volatility tends to spike when liquidity is thin or stressed, and “price movement per unit of traded volume/depth” is a common illiquidity proxy.

Below is how to think about it in crypto specifically, and how you’d use it correctly.

---

## 1) What Yang–Zhang is measuring (and what it isn’t)

**YZ estimates volatility** using:
- “close → next open” returns (overnight in equities),
- “open → close” returns,
- and an intraday range-based component (via Rogers–Satchell).

It’s designed to be robust to drift and to use more information than close-to-close volatility.

**It is not measuring:**
- bid–ask spread
- order book depth
- slippage / market impact
- inventory risk / dealer constraints
- funding stress / margin constraints

Those are the things that usually define “liquidity pressure”.

So YZ ≈ **how much the price moves**, not **how hard it is to trade without moving the price**.

---

## 2) The crypto-specific issue: “overnight” doesn’t exist (24/7 markets)

YZ’s “overnight” term is meaningful in equities because markets close, and the open often contains jumps + repricing.

Crypto trades **24/7**, so you must *invent* a session boundary (e.g., 00:00 UTC daily open). That makes the “close → open” return:

- **dependent on your chosen cut (UTC midnight vs exchange day vs rolling windows)**  
- potentially affected by periodic flows (funding timestamps, daily settlement conventions, Asia/US open overlaps)

This doesn’t make YZ unusable, but it means:

### If you compute daily YZ on crypto:
You’re really computing a **sessionized volatility estimator** whose “overnight” is “last trade before the boundary → first trade after the boundary”.

That boundary can introduce artifacts, so you should:
- keep the boundary consistent (00:00 UTC is common),
- test sensitivity (does your signal change a lot if you shift by 4–8 hours?),
- consider using a volatility estimator that **doesn’t rely on “overnight”** if you want boundary-invariance (e.g., Rogers–Satchell, Garman–Klass, realized volatility).

---

## 3) When it *does* make sense: using YZ as a **price-based stress proxy** inside an LPI

Liquidity pressure often shows up as **large price moves for a given amount of trading activity**.

So a very standard way to turn volatility into a liquidity/pressure proxy is to scale it by activity measures:

### A) “Volatility per unit volume/turnover” (price impact style proxy)
Example constructions (pick one and standardize it):

- **LPI(t) = σ_YZ(t) / DollarVolume(t)**
- **LPI(t) = σ_YZ(t) / √DollarVolume(t)** (often more stable)
- **LPI(t) = σ_YZ(t) / Turnover(t)** (turnover = volume / market cap for spot)
- **LPI(t) = σ_YZ(t) × (1 / Depth(t))** if you have order book depth

Interpretation:
- If volatility rises **without** a commensurate rise in volume/depth, the market is “moving too much for the amount of liquidity” → **liquidity pressure**.

This is closely related in spirit to:
- **Amihud illiquidity** (|return| / dollar volume),
- **Kyle’s lambda** / price impact models (ΔP per signed volume), if you have signed flow.

### B) Composite “liquidity pressure” index (recommended)
If you can access microstructure data, treat YZ as one feature among others.

A simple composite LPI could be a weighted/z-scored combination like:

- z(σ_YZ)  
+ z(Spread)  
+ z(1/Depth)  
+ z(SlippageProxy)  
+ z(|FundingRate|) (for perp-driven markets)  
+ z(OpenInterestChange) (optional)

Then define LPI as a PCA first component or a weighted sum.

YZ contributes: “price stress”
Spreads/depth contribute: “trading frictions”
Funding/OI contribute: “leverage/funding stress” (often a big driver in crypto)

---

## 4) When YZ is a *bad* proxy for liquidity pressure (false positives)

Even if volatility and liquidity stress correlate, you can get big mistakes if you equate them.

YZ (or any volatility) can spike because of:
- genuine information arrival (macro news, protocol exploit news, ETF headlines),
- regime changes / repricing,
- cross-exchange arbitrage gaps closing,
- liquidation cascades (which *are* liquidity-related, but also leverage-related)

And conversely, liquidity can deteriorate while volatility stays muted (e.g., spreads widen but price is range-bound).

So: **high volatility is neither necessary nor sufficient** for liquidity stress.

That’s why the best LPI includes **spread/depth/impact**, not just volatility.

---

## 5) Practical guidance: how to implement this in crypto without fooling yourself

### Step 1 — Decide your data tier
**Tier 1 (best):** order book + trades (L2/L3), spreads, depth, trade signs  
**Tier 2:** trades + best bid/ask (L1), spreads, volumes  
**Tier 3:** OHLCV only

If you’re Tier 3, YZ is *more defensible* as an input because you have little else, but you must admit you’re building a **price-based illiquidity proxy**, not “true liquidity”.

### Step 2 — Choose a bar interval and boundary
- Daily bars: pick 00:00 UTC and stick with it  
- Intraday (e.g., 5m/15m/1h): “open” becomes the first trade of the bar; YZ can still be computed but microstructure noise increases and “overnight” interpretation disappears

For intraday liquidity pressure, many people prefer:
- realized volatility from returns, plus spread/depth measures (if available)

### Step 3 — Normalize and de-seasonalize
Crypto has strong time-of-day and day-of-week seasonality in volume/spreads. If you don’t adjust, your “pressure index” may just become “it’s Sunday”.

Common fixes:
- use rolling z-scores (e.g., 30D or 90D),
- regress out hour-of-day effects for intraday indices,
- compute within-exchange and then aggregate.

### Step 4 — Validate against a ground-truth proxy
If you can, validate your LPI against:
- realized spread / quoted spread,
- slippage of simulated market orders (e.g., $50k / $250k),
- depth at 10bps/25bps,
- price impact coefficients (Kyle lambda).

If your YZ-based LPI doesn’t track these, it’s probably measuring “volatility regime”, not liquidity pressure.

---

## 6) Bottom line

- **Can YZ be used to estimate an LPI in crypto?**  
  **Yes, but only indirectly**—as a **volatility/stress component**, typically scaled by **volume/depth** or combined with **spread/order book** features.

- **Should you use YZ alone as “liquidity pressure”?**  
  **No**—it’s a volatility estimator and will confound information-driven volatility with liquidity-driven stress.

- **Crypto caveat:**  
  Because crypto is **24/7**, the “overnight” part of YZ depends on your chosen session boundary; you should test sensitivity or use estimators that don’t rely on “overnight”.

---

## Two quick questions so I can be more concrete
1) When you say “liquidity pressure index”, do you mean something like **Amihud illiquidity / price impact**, or a **composite stress index** (spread+depth+funding+vol)?  
2) What data do you have access to: **OHLCV only**, **best bid/ask**, or **full order book + trades** (and for CEX, DEX, or both)?

If you answer those, I can propose a specific LPI formula (and a computation recipe) that matches your data and timeframe.
