# Chapter 3 — Risk Management & Portfolio Construction ( \(Z_t, \sigma_t\)  →  Safe Trading Commands)

**Role:** Chief Risk Officer (CRO) & Head of Portfolio Construction  
**Objective:** Convert **raw alpha** (normalized signal \(Z_t \in [-3,3]\)) and **real-time volatility** (\(\sigma_t\), from Chapter 1’s Yang–Zhang pipeline) into **specific, safe, and optimal trading commands** under local-data constraints (1m klines + `bookTicker` liquidity state).

This chapter is intentionally **fail-safe** and **risk-engine-compatible**: we design guardrails so that *even if the signal is wrong*, the system does not behave pathologically.

---

## 3.0 Notation & Contracts (Risk Layer)

We will operate a **single-instrument** portfolio (BTCUSDT perpetual futures) but the design generalizes to multi-asset.

**Inputs available at decision time \(t\):**
- \(Z_t \in [-3,3]\): normalized alpha signal (unitless, signed).
- \(\sigma_t\): volatility estimate (real-time, strictly positive; derived from $\sigma^2_{\mathrm{YZ},t}$).
- `QuoteTick`: top-of-book $(b_t, a_t, q^b_t, q^a_t)$ from `bookTicker`.
- Portfolio: equity \(E_t\), net position \(Q_{\text{actual}}(t)\).

**Outputs of this chapter:**
- \(Q_{\text{target}}(t)\): target base quantity (signed).
- Orders: safe child orders (e.g., TWAP) to move \(Q_{\text{actual}}\to Q_{\text{target}}\), subject to guardrails.

---

## 3.1 Dynamic Position Sizing (Allocation Function)

### 3.1.1 Map normalized signal \(Z_t\) to an expected return $\mathbb{E}[r_t]$

Your \(Z_t\) is bounded but still needs a **unit conversion** into “expected return per decision interval.” We enforce a **saturating** mapping to prevent tail-risk blowups when \(Z_t\) sticks at ±3.

We define:

\[
\mu_t := \mathbb{E}[r_t] = \kappa \cdot \tanh\!\left(\frac{Z_t}{z_{\mathrm{sat}}}\right),
\quad z_{\mathrm{sat}}>0,\ \kappa>0
\]

- \(\kappa\) = “max expected return per decision step” (e.g., per rebalance).
- \(z_{\mathrm{sat}}\) controls how quickly the signal saturates (typical 1.0–2.0).

```python
import math
from decimal import Decimal

# Nautilus types
from nautilus_trader.model.objects import Price, Quantity, Money  # adjust imports if needed

def mu_from_z(z_t: float, kappa: float, z_sat: float = 1.5) -> float:
    # E[r_t] per decision step (unitless return)
    z_clamped = max(-3.0, min(3.0, float(z_t)))
    return float(kappa) * math.tanh(z_clamped / float(z_sat))
```

> **Risk rationale:** even with a bounded \(Z_t\), this prevents the allocation function from becoming overly linear at extremes (a common failure mode in crypto regimes).

---

### 3.1.2 Continuous Kelly / CRRA sizing (core sizing law)

For CRRA risk aversion parameter \(\gamma>0\), the continuous-time Kelly-style allocation fraction is:

\[
f_t^* \propto \frac{\mu_t}{\gamma \sigma_t^2}
\]

We implement it as:

\[
f_t := \mathrm{clip}\!\left(\frac{\mu_t}{\gamma \sigma_t^2},\ -f_{\max},\ +f_{\max}\right)
\]

```python
def frac_from_mu_sigma(mu_t: float, sigma_t: float, gamma: float, f_max: float) -> float:
    if not math.isfinite(mu_t) or not math.isfinite(sigma_t) or sigma_t <= 0.0:
        return 0.0
    raw = mu_t / (float(gamma) * (sigma_t ** 2))
    return max(-float(f_max), min(float(f_max), raw))
```

> **Interpretation:**  
> - Higher expected return \(\mu_t\) increases exposure.  
> - Higher volatility \(\sigma_t\) decreases exposure quadratically.  
> - \(\gamma\) enforces risk aversion.  
> - \(f_{\max}\) caps leverage/overexposure.

---

### 3.1.3 Convert fraction \(f_t\) into a target quantity \(Q^*(t)\)

We treat \(f_t\) as a **target fraction of equity deployed as notional**.

Let:
- \(E_t\) = portfolio equity in quote currency (e.g., USDT),
- \(P_t\) = reference price (we prefer midquote),
- \(N_t = f_t \cdot E_t\) = target notional exposure (signed),
- \(Q^*(t) = \frac{N_t}{P_t}\) = target base quantity (signed, BTC units).

\[
N_t = f_t \, E_t
\qquad\Rightarrow\qquad
Q^*(t) = \frac{N_t}{P_t} = \frac{f_t \, E_t}{P_t}
\]

```python
from decimal import Decimal

def qty_from_fraction_equity_price(f_t: float, equity: Money, ref_price: Price) -> Decimal:
    """
    Returns signed base quantity as Decimal (not Quantity yet).
    We keep sign for portfolio math; Nautilus orders will use side + abs(quantity).
    """
    # Money -> Decimal amount; Price -> Decimal/float.
    # Adjust accessors to your Nautilus version (e.g., equity.as_decimal()).
    E = Decimal(str(equity))          # fallback; prefer equity.as_decimal() if available
    P = Decimal(str(ref_price))       # fallback; prefer ref_price.as_decimal() if available
    if P <= 0:
        return Decimal("0")
    N = Decimal(str(f_t)) * E         # signed notional
    return N / P                      # signed base qty
```

**Reference price \(P_t\):**
- Primary: midquote \(m_t = \frac{a_t+b_t}{2}\).
- Secondary: last bar close (if quote missing).
- Tertiary: fail-safe to flat (if neither available).

---

### 3.1.4 Volatility targeting (optional alternative / secondary cap)

If you prefer a *volatility targeting* upper bound (common CRO overlay), define a maximum position magnitude such that portfolio vol is bounded.

Target portfolio volatility per decision interval: \(\sigma_{\text{target}}\) (in return units).

\[
|Q| \le \frac{\sigma_{\text{target}} \, E_t}{P_t \, \sigma_t}
\]

```python
def vol_target_qty_cap(equity: Money, ref_price: Price, sigma_t: float, sigma_target: float) -> Decimal:
    E = Decimal(str(equity))
    P = Decimal(str(ref_price))
    if not math.isfinite(sigma_t) or sigma_t <= 0.0 or P <= 0:
        return Decimal("0")
    cap = (Decimal(str(sigma_target)) * E) / (P * Decimal(str(sigma_t)))
    return cap.copy_abs()  # magnitude cap
```

**Practical CRO approach:** use Kelly to set direction/magnitude, but apply a volatility cap as a second layer:
- \(Q_{\text{raw}}\) from Kelly,
- cap it by vol targeting and liquidity.

---

### 3.1.5 Instrument constraints (precision, min qty, step size)

Nautilus instruments enforce quantity constraints. We must convert a desired signed Decimal quantity into an executable `Quantity` that respects:

- `size_precision` (decimal places),
- `min_quantity`,
- `step_size` (quantity increment / lot size).

We define step-quantization:

\[
Q_{\text{exec}} = \mathrm{sign}(Q) \cdot \left\lfloor \frac{|Q|}{\Delta q} \right\rfloor \Delta q,
\quad \Delta q = \text{step\_size}
\]

```python
from decimal import Decimal, ROUND_DOWN

def quantize_qty_to_instrument(desired_qty: Decimal, instrument) -> Decimal:
    """
    Returns signed Decimal qty snapped DOWN to step size, and respecting min_quantity.
    Adjust attribute names to your Instrument implementation.
    """
    if desired_qty == 0:
        return Decimal("0")

    sign = Decimal("1") if desired_qty > 0 else Decimal("-1")
    q_abs = desired_qty.copy_abs()

    # --- pull constraints from instrument ---
    # Common fields (names vary by Nautilus version / instrument definition):
    step = Decimal(str(getattr(instrument, "step_size", getattr(instrument, "size_increment", "0.001"))))
    min_q = Decimal(str(getattr(instrument, "min_quantity", "0.001")))
    size_precision = int(getattr(instrument, "size_precision", 3))

    if step <= 0:
        return Decimal("0")

    # Snap to step
    steps = (q_abs / step).to_integral_value(rounding=ROUND_DOWN)
    q_abs = steps * step

    # Enforce precision (defensive)
    q_abs = q_abs.quantize(Decimal("1").scaleb(-size_precision), rounding=ROUND_DOWN)

    if q_abs < min_q:
        return Decimal("0")

    return sign * q_abs
```

---

### 3.1.6 Strategy method: `calculate_target_position(signal_strength, volatility)`

This is the **single canonical function** the rest of the trading system should call.

Key CRO requirements built in:
- **Fail-safe on NaN/Inf volatility:** target becomes flat (0).
- **Fail-safe on missing instrument:** stop strategy (fail-fast).
- **Quantization:** final target is executable.

```python
import math
from decimal import Decimal
from nautilus_trader.trading.strategy import Strategy

from nautilus_trader.model.objects import Price, Quantity, Money  # adjust imports if needed
from nautilus_trader.model.enums import OrderSide  # for later reconciliation

class AlphaStrategy(Strategy):
    # ... your Chapter 2 init ...
    # Assumes:
    #   self.instrument_id
    #   self.liq.mid (or quote cache)
    #   self.config has gamma, kappa, f_max, sigma_target, etc.

    def calculate_target_position(self, signal_strength: float, volatility: float) -> Decimal:
        """
        Returns signed target base quantity as Decimal (e.g., BTC).
        Order generation will convert this into (OrderSide, Quantity(abs)).
        """
        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            # Fail-fast: running without instrument contract is unsafe.
            self.log.error(f"Instrument missing: {self.instrument_id} -> stopping strategy")
            self.stop()
            return Decimal("0")

        # --- Fail-safe: volatility must be finite and > 0 ---
        if (not math.isfinite(volatility)) or volatility <= 0.0:
            # Fail-safe posture: flatten (or reduce-only)
            self.log.warning(f"Invalid volatility={volatility} -> target=0 (fail-safe)")
            return Decimal("0")

        # 1) Map Z -> mu
        mu_t = mu_from_z(
            z_t=float(signal_strength),
            kappa=float(self.config.alpha_kappa),   # you define in config
            z_sat=float(self.config.alpha_z_sat),   # you define in config
        )

        # 2) Kelly fraction
        f_t = frac_from_mu_sigma(
            mu_t=mu_t,
            sigma_t=float(volatility),
            gamma=float(self.config.risk_gamma),
            f_max=float(self.config.f_max),
        )

        # 3) Reference price: prefer midquote
        ref_price = self._get_reference_price()
        if ref_price is None:
            # No price => cannot size safely
            self.log.warning("No reference price available -> target=0 (fail-safe)")
            return Decimal("0")

        # 4) Equity in quote currency (USDT)
        equity = self._get_equity_money()
        if equity is None:
            self.log.warning("Equity unavailable -> target=0 (fail-safe)")
            return Decimal("0")

        # 5) Raw target qty (signed)
        q_raw = qty_from_fraction_equity_price(f_t=f_t, equity=equity, ref_price=ref_price)

        # 6) Optional vol-target cap
        if getattr(self.config, "sigma_target", None) is not None:
            cap = vol_target_qty_cap(
                equity=equity,
                ref_price=ref_price,
                sigma_t=float(volatility),
                sigma_target=float(self.config.sigma_target),
            )
            if cap > 0:
                # clamp magnitude
                if q_raw.copy_abs() > cap:
                    q_raw = cap.copy_sign(q_raw)

        # 7) Quantize to instrument constraints
        q_exec = quantize_qty_to_instrument(q_raw, instrument)

        return q_exec

    def _get_reference_price(self) -> Price | None:
        # Primary: last quote mid
        qt = self.cache.quote_tick(self.instrument_id)
        if qt is not None:
            bid = float(qt.bid_price)
            ask = float(qt.ask_price)
            if bid > 0 and ask > 0 and ask >= bid:
                mid = 0.5 * (bid + ask)
                return Price.from_str(str(mid))

        # Secondary: last bar close
        bar = self.cache.bar(self.bar_type_1m)  # adjust accessor; may require (bar_type, instrument_id)
        if bar is not None:
            c = float(bar.close)
            if c > 0:
                return Price.from_str(str(c))

        return None

    def _get_equity_money(self) -> Money | None:
        # Adjust to your account/portfolio wiring.
        # Many setups expose equity via portfolio/account objects.
        try:
            # Example placeholder:
            return self.portfolio.equity()  # type: ignore[attr-defined]
        except Exception:
            return None
```

---

## 3.2 Nautilus Risk Engine Configuration (Hard Reject Layer)

Your strategy must be **rejected** by the Risk Engine if it behaves anomalously. The Risk Engine is the *last line of defense*; your strategy still implements pre-trade checks to avoid repeated rejects.

### 3.2.1 Derive `max_order_qty` from average top-of-book liquidity

Let \(L_t := \min(q^b_t, q^a_t)\) be the instantaneous “top size” liquidity proxy from `bookTicker`.  
Let \(\bar{L}\) be a robust average (median or trimmed mean) over a calibration window.

We set:

\[
Q_{\max,\text{order}} = \rho \, \bar{L}
\quad\text{with}\quad 0 < \rho \ll 1
\]

```python
from decimal import Decimal

def max_order_qty_from_liquidity(L_bar: Decimal, rho: Decimal) -> Quantity:
    # max_order_qty = rho * L_bar
    q = (rho * L_bar)
    return Quantity.from_str(str(q))
```

**CRO guidance for \(\rho\):** start conservative:
- \(\rho = 0.01\) to \(0.05\) for aggressive markets (crypto) depending on slippage tolerance.

> In practice: compute \(\bar{L}\) offline from your local `bookTicker` CSVs (streaming median/quantiles), then hardcode into config per instrument and epoch.

---

### 3.2.2 Derive `max_notional_per_order` from equity

We cap per-order notional as a fixed fraction of equity.

\[
N_{\max,\text{order}} = \eta \, E_t
\quad\text{with}\quad 0 < \eta < 1
\]

```python
def max_notional_per_order_from_equity(equity: Money, eta: float) -> Money:
    # N_max = eta * equity
    E = Decimal(str(equity))
    N = Decimal(str(eta)) * E
    # If Money.from_str isn't available in your version, construct Money via (amount, currency).
    return Money.from_str(f"{N} {equity.currency}")  # adjust accessor
```

**Typical CRO value:** \(\eta = 0.1\%\) to \(1.0\%\) for single-order caps (smaller if using TWAP slices).

---

### 3.2.3 Daily loss circuit breaker: `max_daily_loss`

Hard circuit breaker: if realized PnL for the day breaches a threshold, reject new risk-increasing orders.

\[
L_{\max,\text{day}} = \delta \, E_0
\quad\text{with}\quad 0 < \delta < 1
\]

```python
def max_daily_loss_from_starting_equity(starting_equity: Money, delta: float) -> Money:
    E0 = Decimal(str(starting_equity))
    L = Decimal(str(delta)) * E0
    return Money.from_str(f"{L} {starting_equity.currency}")  # adjust accessor
```

**Typical CRO value:** \(\delta = 1\%\) to \(3\%\) depending on strategy Sharpe + tail profile.

---

### 3.2.4 TradingNode instantiation with a concrete `RiskEngineConfig`

Below is a **code-spec** instantiation. The exact import paths and field names can vary across NautilusTrader versions, but the configuration intent is invariant.

```python
from decimal import Decimal

from nautilus_trader.config import (
    TradingNodeConfig,
    RiskEngineConfig,
    LoggingConfig,
    PortfolioConfig,
)
from nautilus_trader.trading.node import TradingNode

from nautilus_trader.model.objects import Quantity, Money

# --- Example calibrated constants (replace with offline estimates) ---
L_bar = Decimal("5.0")          # median(min(bid_size, ask_size)) in BTC units from bookTicker
rho = Decimal("0.02")           # 2% of top-of-book median
max_order_qty = Quantity.from_str(str(rho * L_bar))  # e.g., 0.10 BTC

starting_equity = Money.from_str("100000 USDT")      # replace with your actual account equity
eta = 0.005                                          # 0.5% per order notional cap
delta = 0.02                                         # 2% daily loss circuit breaker

max_notional_per_order = Money.from_str(f"{Decimal(str(eta)) * Decimal('100000')} USDT")
max_daily_loss = Money.from_str(f"{Decimal(str(delta)) * Decimal('100000')} USDT")

risk_config = RiskEngineConfig(
    max_order_qty=max_order_qty,
    max_notional_per_order=max_notional_per_order,
    max_daily_loss=max_daily_loss,

    # Additional hardening knobs (names vary by version):
    # reject_orders=True,
    # max_open_orders=20,
    # max_position_qty=Quantity.from_str("2.0"),
)

node_config = TradingNodeConfig(
    logging=LoggingConfig(level="INFO"),
    portfolio=PortfolioConfig(
        # account_id, base_currency, etc. depend on your wiring
    ),
    risk_engine=risk_config,
    # execution_engine=..., data_engine=..., etc.
)

node = TradingNode(config=node_config)
node.run()
```

**CRO note:** even if the Risk Engine lacks a native `max_daily_loss`, you must still implement a *strategy-level* circuit breaker (Section 3.4) because daily PnL semantics vary (mark-to-market vs realized vs fees).

---

## 3.3 Execution Guardrails & Liquidity Gates (Pre-Trade Checks)

We now impose **market-state gating** using only top-of-book (`bookTicker`) plus rolling spread statistics.

### 3.3.1 Spread definition from quotes

Let bid \(b_t\), ask \(a_t\). Define spread:

\[
S_t = a_t - b_t
\]

```python
def spread_from_quote(bid: Price, ask: Price) -> float:
    S_t = float(ask) - float(bid)
    return S_t
```

We maintain a rolling spread mean and std for anomaly detection.

---

### 3.3.2 Rolling spread threshold gate → REDUCING mode

We define a rolling baseline \(\bar{S}\) and rolling standard deviation \(\sigma_S\).  
If spread is abnormally wide:

\[
S_t > \bar{S} + k \, \sigma_S
\quad\Rightarrow\quad \text{TradingState} = \text{REDUCING}
\]

```python
import math
from collections import deque

class RollingSpreadStats:
    def __init__(self, window: int) -> None:
        self.window = int(window)
        self.xs = deque(maxlen=self.window)
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, x: float) -> None:
        if len(self.xs) == self.xs.maxlen:
            old = self.xs[0]
            self.sum -= old
            self.sumsq -= old * old
        self.xs.append(x)
        self.sum += x
        self.sumsq += x * x

    def mean_std(self) -> tuple[float, float]:
        n = len(self.xs)
        if n < 2:
            return (math.nan, math.nan)
        mean = self.sum / n
        var = max(0.0, (self.sumsq / n) - mean * mean)
        return (mean, math.sqrt(var))
```

And in the strategy:

```python
from nautilus_trader.model.enums import TradingState  # adjust import if needed

class AlphaStrategy(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.spread_stats = RollingSpreadStats(window=int(config.spread_window))
        self.trading_state = TradingState.ACTIVE  # or NORMAL, depending on enum

    def _update_liquidity_gate(self) -> None:
        qt = self.cache.quote_tick(self.instrument_id)
        if qt is None:
            # No quote => cannot assess liquidity => fail-safe reducing
            self.trading_state = TradingState.REDUCING
            return

        bid = float(qt.bid_price)
        ask = float(qt.ask_price)
        if bid <= 0 or ask <= 0 or ask < bid:
            self.trading_state = TradingState.REDUCING
            return

        S_t = ask - bid
        self.spread_stats.update(S_t)

        S_bar, S_std = self.spread_stats.mean_std()
        if not math.isfinite(S_bar) or not math.isfinite(S_std):
            # Not warmed up => do not trust stats => be conservative
            self.trading_state = TradingState.REDUCING
            return

        k = float(self.config.spread_sigma_k)
        if S_t > (S_bar + k * S_std):
            self.trading_state = TradingState.REDUCING
        else:
            self.trading_state = TradingState.ACTIVE
```

**What REDUCING means (CRO policy):**
- **No risk-increasing trades.**
- Only allow orders that move the absolute position toward zero.

A strict and simple implementation is: when REDUCING, force \(Q_{\text{target}}(t)=0\). That ensures only flattening actions.

---

### 3.3.3 Access the latest quote via Cache (`self.cache.quote_tick(...)`)

Required access pattern:

```python
qt = self.cache.quote_tick(self.instrument_id)
if qt is not None:
    bid = float(qt.bid_price)
    ask = float(qt.ask_price)
    spread = ask - bid
```

**Additional CRO hardening (recommended):**
- Reject trading if quote is stale relative to decision clock (in live).
- Reject trading if `bid_size`/`ask_size` are below a minimum (already in Chapter 2 liquidity state).

---

## 3.4 Portfolio State & Reconciliation (Target vs Actual)

### 3.4.1 State tracking: Portfolio vs Cache

You must distinguish between:
- `self.portfolio.net_position(instrument_id)` — **portfolio-level net position** (fast, aggregated, what you should use for control decisions).
- `self.cache.position(position_id)` — **position object** with richer details (fills, entry price, realized PnL, etc.) but requires the correct `position_id`.

**CRO rule:** use `net_position` for the control loop; use `cache.position(...)` only for audit/debug/reconciliation detail.

---

### 3.4.2 Reconciliation equation & threshold

Compute the delta:

\[
\Delta Q(t) = Q_{\text{target}}(t) - Q_{\text{actual}}(t)
\]

```python
from decimal import Decimal

def delta_qty(q_target: Decimal, q_actual: Decimal) -> Decimal:
    # ΔQ = Q_target - Q_actual
    return q_target - q_actual
```

Only trade if the delta is large enough:

\[
|\Delta Q(t)| > \varepsilon_Q
\]

```python
def should_rebalance(delta_q: Decimal, eps_q: Decimal) -> bool:
    return delta_q.copy_abs() > eps_q
```

Where \(\varepsilon_Q\) is typically:
- at least one `step_size`,
- often several steps to avoid churn (especially under spread widening).

---

### 3.4.3 Control loop (runs on `on_bar` or a Timer)

**You can run this loop**:
- on every 1m bar (`on_bar`) if you want low churn, or
- on a fixed timer (e.g., every 1–5 seconds) if you’re closer to HFT execution.

#### ASCII — Reconciliation Control Loop

```
┌───────────────────────────┐
│ Timer / on_bar event       │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 1) Update liquidity gate   │
│    - read QuoteTick        │
│    - compute spread stats  │
│    - set ACTIVE/REDUCING   │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 2) Compute Q_target        │
│    - if REDUCING -> 0      │
│    - else Kelly/vol target │
│    - quantize to step      │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 3) Read Q_actual           │
│    self.portfolio.net_pos  │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 4) ΔQ = Q_target - Q_actual│
│    if |ΔQ| <= eps -> no-op │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ 5) SubmitOrder (guarded)   │
│    - side from sign(ΔQ)    │
│    - qty = abs(ΔQ)         │
│    - exec algo (TWAP)      │
└───────────────────────────┘
```

---

### 3.4.4 Implementation: guarded rebalance method (order emission)

```python
from decimal import Decimal
import math

from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model import ExecAlgorithmId
from nautilus_trader.model.objects import Quantity

class AlphaStrategy(Strategy):

    def on_bar(self, bar) -> None:
        # Example: rebalance on bar close
        self.reconcile_and_trade()

    def reconcile_and_trade(self) -> None:
        # --- 0) Liquidity gate update ---
        self._update_liquidity_gate()

        # --- 1) Compute target (fail-safe aware) ---
        if self.trading_state == TradingState.REDUCING:
            q_target = Decimal("0")
        else:
            # These should be current values computed in your alpha layer
            z_t = float(getattr(self, "alpha_z", 0.0))
            sigma_t = float(getattr(self, "sigma", float("nan")))
            q_target = self.calculate_target_position(signal_strength=z_t, volatility=sigma_t)

        # --- 2) Read actual net position ---
        # Adjust based on your Nautilus portfolio API. We want signed base qty.
        net_pos = self.portfolio.net_position(self.instrument_id)  # returns Position or Quantity depending on version
        q_actual = Decimal(str(net_pos))  # adapt: net_pos.quantity, net_pos.as_decimal(), etc.

        # --- 3) Delta and threshold ---
        dQ = q_target - q_actual

        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            self.stop()
            return

        step = Decimal(str(getattr(instrument, "step_size", getattr(instrument, "size_increment", "0.001"))))
        eps = Decimal(str(getattr(self.config, "rebalance_epsilon_qty", step)))

        # Fail-fast: if anything is not finite, do nothing (and optionally stop)
        if not math.isfinite(float(dQ)):
            self.log.error(f"Non-finite ΔQ computed: {dQ} -> no trade (fail-safe)")
            return

        if dQ.copy_abs() <= eps:
            return  # no-op

        # --- 4) Convert ΔQ into an order (side + abs qty) ---
        side = OrderSide.BUY if dQ > 0 else OrderSide.SELL
        qty_abs = quantize_qty_to_instrument(dQ.copy_abs(), instrument)
        if qty_abs <= 0:
            return

        order_qty = Quantity.from_str(str(qty_abs))

        # --- 5) Submit via execution algorithm (preferred) ---
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=order_qty,
            exec_algorithm_id=ExecAlgorithmId("TWAP_ADAPTIVE"),
        )

        self.submit_order(order)
```

---

## 3.5 Fail-Safe Design (NaN / Inf Volatility & “Fail-Fast” Behavior)

The most dangerous failure mode in an automated strategy is *silently producing invalid sizing* (NaN/Inf) that becomes a gigantic order after type conversion.

### 3.5.1 Fail-safe rule for volatility

If \(\sigma_t\) is invalid:

- \(\sigma_t \notin \mathbb{R}\) (NaN/Inf), or
- \(\sigma_t \le 0\),

then the system must enter a safe posture.

Operationally we enforce:

- \(Q_{\text{target}}(t) := 0\) (flatten), and/or
- `TradingState.REDUCING`, and/or
- stop strategy (`self.stop()`) if the condition persists beyond a grace period.

This is consistent with a **fail-fast architecture**: it is better to stop than to trade with undefined risk.

**Implementation already included** in `calculate_target_position()`:
- invalid volatility returns `Decimal("0")` and logs a warning.

---

## 3.6 Risk Control Flow (End-to-End ASCII)

```
┌──────────────────────────────┐
│ Alpha Layer (Chapter 1/2)     │
│  - Z_t in [-3,3]              │
│  - σ_t from YZ volatility     │
│  - QuoteTick liquidity state  │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ Sizing Model (3.1)            │
│  μ_t = κ tanh(Z_t/z_sat)      │
│  f_t = μ_t/(γ σ_t^2)          │
│  Q_raw = f_t * E / P          │
│  quantize -> Q_target         │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ Pre-Trade Checks (3.3)        │
│  - spread anomaly gate        │
│  - low-liquidity gate         │
│  - NaN/Inf volatility gate    │
│  => ACTIVE or REDUCING        │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ Reconciliation (3.4)          │
│  ΔQ = Q_target - Q_actual     │
│  if |ΔQ| > ε_Q -> trade       │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ Order Generation              │
│  side = sign(ΔQ)              │
│  qty  = abs(ΔQ) quantized     │
│  submit (TWAP_ADAPTIVE)       │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ Risk Engine (3.2) HARD REJECT │
│  - max_order_qty              │
│  - max_notional_per_order     │
│  - max_daily_loss             │
└──────────────────────────────┘
```

---

## What you should provide next (so Chapter 3 can be *fully exact* in your repo)

To make the code “drop-in exact” (no placeholder accessors), I need **your NautilusTrader version** and the exact APIs for:
1. `Money` construction and `portfolio.equity()` accessor
2. `instrument.step_size / min_quantity / size_precision` actual attribute names
3. `portfolio.net_position(...)` return type and how to extract signed quantity

If you paste:
- your `pip freeze | grep nautilus` (or `pyproject.toml`),
- and a short snippet showing how you currently read equity and position,

I will rewrite the code in this chapter into **fully exact imports + exact field names** for your environment (no “adjust imports” comments).
