"""
Foundation Strategy — Kalshi Prediction Markets
================================================
Applies two techniques from finance lecture notes to prediction markets:

1. Return Predictability (Lecture 5.6):
   A signal forecasts expected returns. Here: OLS regression slope over
   a lookback window estimates the expected daily price change (μ̂).
   This is the prediction-market analog of the dividend yield forecasting
   regression — a known signal used to forecast future returns.

2. Volatility Timing (Lecture 5.7):
   Scale positions by inverse realized variance (1/RV). Invest more when
   volatility is low, less when it is high. Realized variance is computed
   from daily price changes over the same lookback window.

Combined mean-variance optimal weight (both lectures together):
    w = μ̂ / (γ · RV)

where:
    μ̂   = OLS-forecasted annualized expected price change  (Lecture 5.6)
    RV   = annualized realized variance of daily changes    (Lecture 5.7)
    γ    = risk aversion parameter (default: 2)


DESIGN CHOICES
==============

1. Signal: OLS slope of price changes
   The lecture uses dividend yield to predict future stock returns — a
   signal known at time t that forecasts returns over the next period.
   The analog here is the OLS slope of recent daily price changes: it
   answers "is this market trending, and in which direction?" A positive
   slope means prices have been drifting toward YES resolving. That is
   the forecast μ̂. Annualizing by 252 is for dimensional consistency
   with RV — since both scale the same way, it cancels in the weight
   formula and only their ratio matters.

2. Why OLS and not something else
   The lecture explicitly uses OLS regression as the forecasting tool
   and R² as the quality filter. The alternative was the short/long
   window momentum composite from backtest.py, but that adds complexity
   without grounding in either lecture. OLS slope is the simplest
   regression-based forecast.

3. The R² filter
   Comes directly from Lecture 5.6. A low R² means the regression has
   no explanatory power and the slope is noise, not a forecast. Default
   is 0.10 — kept deliberately loose because prediction market price
   series are short and noisy. 0.25 generated zero trades; 0.15 generated
   too few (14) to be meaningful. 0.10 is the working threshold.
   Tighten with --min-r2 to trade less but with higher signal confidence.

4. Realized variance
   Lecture 5.7 computes RV from daily equity returns aggregated to
   monthly. Here: variance of daily price changes over the lookback
   window, annualized by ×252. The key adaptation: equity returns are
   log-returns of prices. Here prices are already probabilities in (0,1),
   so log-returns don't make sense. Daily price changes (first
   differences) are the natural unit. Their variance measures how noisy
   this market's price discovery has been recently.

5. The weight formula: w = μ̂ / (γ · RV)
   This is the core of both lectures combined. Lecture 5.6 says invest
   proportionally to expected return. Lecture 5.7 says scale inversely
   to variance. The mean-variance formula does both simultaneously — it
   is not two separate rules bolted together, it is the single formula
   that falls out of mean-variance optimization when both μ̂ and RV vary.
   The sign of w determines direction: positive → BUY YES, negative →
   BUY NO.

6. The cap on weights and choice of γ
   Lecture 5.7 explicitly warns: "Leverage can get extreme! Weights can
   exceed 10x during calm periods." For prediction markets this is worse
   — when RV is very low (a stable market near 50%) and slope is nonzero,
   μ̂/RV blows up. The cap at ±0.25 prevents the formula from recommending
   300% of bankroll on a single market.

   γ=2 causes the cap to bind on virtually every trade — all weights are
   clipped to ±0.25, so vol-timing from Lecture 5.7 produces no position
   differentiation. γ=20 was tested to fix this: weights became
   differentiated, but performance worsened. The reason: prediction
   markets are backwards from equities. High-vol markets had the BEST
   mean-reversion (75% win rate, highest P&L), so down-weighting them
   via inverse-variance scaling actively hurt returns. Lecture 5.7's
   logic does not apply in this direction for this asset class.
   γ=2 is kept so the cap binds uniformly — effectively equal-sizing all
   trades, which empirically outperforms vol-scaled sizing here.

7. The scaling constant
   Directly from Lecture 5.7: normalize so the average absolute weight
   equals 1, making the strategy comparable to a naive equal-weighted
   baseline. The lecture does this with c = 1 / mean(1/RV) on a single
   factor; here it is applied across the cross-section of markets.

8. Backtest P&L formula: w * (exit - entry)
   w * ΔP is dollar gain per dollar of bankroll. If w = 0.20 and price
   moves from 0.50 to 0.55, profit is 0.20 × 0.05 = 1¢ per dollar. The
   spread cost deducts half the bid-ask spread at entry. This is the same
   structure as backtest.py but generalized: instead of a fixed binary
   bet size, position size is the continuous MV weight.

9. Vol quintile table in backtest report
   Replicates the Lecture 5.7 empirical chart: sort observations by
   realized variance and ask whether mean net P&L declines as RV rises.
   If yes, the vol-timing mechanism is working. If no, the RV scaling
   is not helping and should be investigated.

10. Sports market exclusion
    Backtesting showed every sports market (NBA, Premier League) was a
    consistent loser while political and financial markets were profitable.
    The economic reason: sports prices follow momentum — a team on a
    winning streak stays hot, so recent price trends continue rather than
    reverse. Political markets are driven by polling and news cycles,
    which mean-revert. The mean-reversion signal is structurally wrong
    for sports. Sports tickers are excluded by keyword before any history
    is fetched.

11. Signal direction and trade side
    The OLS slope was initially used as μ̂ directly (momentum: bet that
    trends continue). Backtesting showed a 31% win rate — the signal was
    directionally wrong. Prediction market prices mean-revert; equity
    returns trend. The slope is negated so the strategy bets against
    recent trends rather than with them.

    Backtesting also showed a persistent asymmetry: BUY NO trades had a
    positive net Sharpe (+0.67) while BUY YES trades were consistently
    negative (-1.10). This likely reflects that markets trending upward
    are more prone to overreaction and reversion than markets trending
    downward. The strategy was restricted to BUY NO only based on this
    empirical finding.

12. What was deliberately left out
    HAC standard errors (Lecture 5.6): needed for overlapping multi-year
    return windows. With a 1-day hold period there is no meaningful
    overlap, so they are not needed here.

    Out-of-sample R² vs. historical mean (Lecture 5.6): the key
    evaluation tool for equity predictability. Does not translate because
    each prediction market is unique — there is no stable unconditional
    mean to benchmark against.

    Kelly sizing from backtest.py: replaced entirely by the MV weight
    formula. Kelly and MV are related (Kelly ≈ MV with γ = 1/wealth),
    but the MV formula is what both lectures derive, so it is used
    directly here.

13. Category expansion (all events, not just Politics)
    The initial version fetched only the "Politics" category. This
    produced 96 trades across 42 markets in a 180-day backtest — too
    few for reliable inference (bootstrap Sharpe CI crossed zero at the
    5% level). Expanding to all categories while keeping the sports
    keyword exclusion is the cleanest way to increase sample size
    without changing any strategy parameters. In practice the top-200
    by volume remains dominated by political markets, so this is mostly
    a quality-of-life change that future-proofs the fetcher as Kalshi
    adds new market types.

14. Minimum entry price filter (--min-price, default 0.0)
    Initial backtest (96 trades, no price filter) showed that the
    38 trades with entry price below 0.15 had a negative Sharpe
    (-0.34, ns). The economic reason: a BUY NO on a market already
    priced near zero (e.g. P=0.08) is betting that a near-certain NO
    will resolve NO — there is almost no price movement left to capture,
    and any upward noise produces an outsized loss relative to the tiny
    potential gain. The mean-reversion signal is structurally weakest
    for markets already near their floor.

    After filtering to P ≥ 0.15, the backtest (58 trades) showed:
      Win rate   : 54% → 64%
      Sharpe     : +1.12 → +1.54
      Bootstrap CI Sharpe: [-0.07, +2.43] → [+0.01, +3.37]
    The lower bound of the bootstrap CI crossed into positive territory,
    meaning the strategy now clears zero at 95% bootstrap confidence.

    A further sub-sample analysis revealed a sharp non-linearity:

      Entry price range    n    Win%    Sharpe    Significance
      ─────────────────   ──   ─────   ──────    ────────────
      Low   (P < 0.31)    22    68%    +1.46     ns
      Mid   (0.31–0.56)   13    23%    −2.32     ns
      High  (P ≥ 0.56)    23    83%    +4.88     ***

    High-entry trades are by far the strongest signal: a market priced
    above 56% has already drifted well toward YES, making it the purest
    mean-reversion bet. Mid-range markets (near equilibrium) are the
    worst — no clear momentum to fade and no floor/ceiling pressure.
    The --min-price flag encodes the lower bound of this finding; a
    --max-price flag could further isolate the high-entry bucket.

15. Empirical analysis: what the backtest report tests and why
    The backtest report includes three layers of analysis beyond the
    headline Sharpe and win rate:

    Empirical results (t-test, CI, bootstrap Sharpe):
      A point estimate of Sharpe is almost meaningless at 50–100 trades.
      The t-test tests whether mean net PnL is distinguishable from zero.
      The 95% CI shows the range of plausible mean PnL values. The
      bootstrap Sharpe CI (2,000 resamples) is reported separately
      because Sharpe's sampling distribution is not Normal, especially
      at small n. Significance stars follow the p=0.10/0.05/0.01
      convention. After the price filter, the strategy sits at p≈0.056
      (borderline) with a bootstrap Sharpe CI that barely clears zero.

    Sub-sample analysis (temporal, R², entry price):
      Temporal split: the first half of trades had SR=+2.93 (**) while
      the second half had SR=+0.99 (ns). This degradation persists after
      the price filter and is the main unresolved concern — it could
      reflect genuine signal decay, survivorship in the second half, or
      simply small-sample noise. R² split: counterintuitively, low-R²
      trades outperform high-R² trades in both runs. A tighter OLS fit
      does not predict better outcomes, possibly because high-R² markets
      are genuinely trending (the signal is real momentum, not noise to
      fade). Entry price split: documented in note 14 above.

    Robustness checks (R² sensitivity, spread cost, leave-one-out):
      Tightening min-R² above 0.15 collapses the trade count to near
      zero, confirming the strategy has no headroom on signal quality.
      Gross vs. net shows the edge is real before costs (p=0.025, **)
      but spread drag is material (≈17% of gross PnL). Leave-one-out
      drops the single most-traded ticker and checks whether results
      survive — after the price filter, dropping the top market actually
      improved significance (p=0.05, **), indicating the LOO market was
      a drag, not a driver.

    Concentration caveat: across both backtest runs, two Texas Senate
    nomination markets (KXTXSENCOMBO-26NOV-TALCOR, KXSENATETXR-26-JC)
    contributed approximately 100% or more of total net PnL, with the
    remainder of the portfolio roughly flat. This is the primary open
    risk: the strategy's realized profitability may be specific to those
    markets rather than a general phenomenon. More markets and a longer
    history are needed to assess generalizability.


Usage:
    python model.py                                # live screener
    python model.py --backtest                     # walk-forward backtest
    python model.py --backtest --hold 3            # 3-day hold period
    python model.py --backtest --min-price 0.15    # filter low-price trades
    python model.py --backtest --plot               # save PNG plots to disk
    python model.py --gamma 3 --lookback 20 --min-volume 50
    python model.py --export results.csv

Flags:
    --gamma       Risk aversion γ (default: 2)
    --lookback    Days of price history for signal + RV (default: 30)
    --hold        Hold period in days for backtest (default: 7)
    --min-volume  Minimum market volume (default: 50)
    --max-weight  Maximum raw position size before normalization (default: 0.25)
    --min-r2      Minimum R² to trust the signal (default: 0.10)
    --min-price   Skip trades where entry price < this threshold (default: 0.0)
    --top-n       Positions to display (default: 20)
    --fetch-days  Days of history to fetch per market in backtest (default: 180)
    --limit       Max markets to process, ranked by volume (default: 200)
    --export      Save results to CSV
    --plot        Save backtest plots as PNG (backtest only)
    --backtest    Run walk-forward backtest instead of live screener

Requirements:
    pip install requests pandas scipy numpy tabulate colorama matplotlib
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
import pandas as pd
from scipy import stats
from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS  = {"Accept": "application/json"}

# ── Defaults ──────────────────────────────────────────────────────────────────
GAMMA       = 2.0   # risk aversion — vol-timing inverted for prediction markets (see note 6)
LOOKBACK    = 30    # days of price history for estimation
MIN_R2      = 0.10  # minimum R² to trust the OLS signal
MAX_WEIGHT  = 0.25  # cap on raw weight before normalization
MIN_HISTORY = 10    # minimum data points required to estimate anything
MAX_SPREAD  = 0.10  # skip markets where bid-ask spread > 10% of mid-price

# Sports markets follow momentum (teams on winning streaks stay hot), not
# mean-reversion. Backtest showed every sports market was a consistent loser
# while political/financial markets were consistently profitable.
SPORTS_KEYWORDS = ["nba", "nfl", "nhl", "mlb", "nascar", "pga", "premier",
                   "champions", "bundesliga", "laliga", "seriea", "ligue1",
                   "mvp", "championship", "playoff", "superbowl", "worldseries"]


# ── API helpers ───────────────────────────────────────────────────────────────

def kalshi_get(path: str, params: dict = {}) -> dict:
    url = f"{BASE_URL}/{path.lstrip('/')}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def to_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if 0.0 <= f <= 1.0 else None
    except (TypeError, ValueError):
        return None


def fetch_events() -> list[dict]:
    events, cursor, page = [], None, 0
    print(f"{Fore.YELLOW}Fetching all open events…{Style.RESET_ALL}", flush=True)
    while True:
        params = {"limit": 100, "status": "open", "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor
        data   = kalshi_get("events", params)
        batch  = data.get("events", [])
        events.extend(batch)
        page  += 1
        cursor = data.get("cursor")
        print(f"\r  Page {page}: {len(events)} events…", end="", flush=True)
        if not cursor or not batch:
            break
        time.sleep(0.2)
    print(f"\r  Done — {len(events)} events fetched.          ")
    return events


def extract_markets(events: list[dict]) -> list[dict]:
    markets = []
    for ev in events:
        for m in ev.get("markets", []):
            bid  = to_float(m.get("yes_bid_dollars"))
            ask  = to_float(m.get("yes_ask_dollars"))
            last = to_float(m.get("last_price_dollars"))
            vol  = float(m.get("volume_fp") or 0)
            if vol > 0 and (bid or ask or last):
                m["_event_title"] = ev.get("title", "")
                markets.append(m)
    return markets


def fetch_market_history(ticker: str, event_ticker: str, days: int) -> list[float]:
    """Return list of daily close prices in (0, 1)."""
    try:
        series_ticker = event_ticker.split("-")[0] if event_ticker else ""
        if not series_ticker:
            return []
        end_ts   = int(time.time())
        start_ts = end_ts - 86400 * days
        data = kalshi_get(
            f"series/{series_ticker}/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": 1440},
        )
        prices = []
        for c in data.get("candlesticks", []):
            close = to_float((c.get("price") or {}).get("close_dollars"))
            if close is not None and 0 < close < 1:
                prices.append(close)
        return prices
    except Exception:
        return []


def midpoint(m: dict) -> Optional[float]:
    bid = to_float(m.get("yes_bid_dollars"))
    ask = to_float(m.get("yes_ask_dollars"))
    if bid is not None and ask is not None and ask > 0:
        return (bid + ask) / 2
    return to_float(m.get("last_price_dollars"))


def spread_pct(m: dict) -> float:
    bid = to_float(m.get("yes_bid_dollars"))
    ask = to_float(m.get("yes_ask_dollars"))
    if not bid or not ask or ask == 0:
        return 0.0
    mid = (bid + ask) / 2
    return (ask - bid) / mid if mid else 0.0


def days_to_close(m: dict) -> Optional[int]:
    for field in ("close_time", "expiration_time", "latest_expiration_time"):
        val = m.get(field)
        if val:
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                return max(0, (dt - datetime.now(timezone.utc)).days)
            except Exception:
                pass
    return None


# ── Signal and variance estimation ────────────────────────────────────────────

def estimate(prices: list[float]) -> Optional[dict]:
    """
    From a price history, estimate:

      μ̂  — OLS slope of price changes over time, annualized.
            This is the Lecture 5.6 signal: a regression-based forecast
            of the expected return going forward.

      RV  — Realized variance of daily price changes, annualized.
            This is the Lecture 5.7 input: recent variance used to scale
            the position inversely.

    Returns None if there is insufficient data.
    """
    n = len(prices)
    if n < MIN_HISTORY:
        return None

    changes = np.diff(np.array(prices, dtype=float))  # daily price changes
    if len(changes) < 2:
        return None

    # ── Lecture 5.6: OLS forecast of expected daily price change ──────────
    x = np.arange(len(changes), dtype=float)
    slope, _, r_val, _, _ = stats.linregress(x, changes)
    r2 = r_val ** 2

    # ── Lecture 5.7: Realized variance of daily price changes ─────────────
    rv = float(np.var(changes, ddof=1)) * 252   # annualized
    if rv < 1e-8:
        return None

    return {
        "mu_hat": -slope * 252,  # negated: bet on mean-reversion, not momentum
        "rv":     rv,            # annualized realized variance
        "r2":     r2,
        "n":      n,
    }


# ── Weight formula (both lectures combined) ───────────────────────────────────

def mv_weight(mu_hat: float, rv: float, gamma: float, max_weight: float) -> float:
    """
    Mean-variance optimal weight: w = μ̂ / (γ · RV)

    This unifies both lectures:
      - Numerator (μ̂): signal from Lecture 5.6 — invest more when expected
        return is high.
      - Denominator (γ · RV): from Lecture 5.7 — scale down when variance
        is high.

    Capped at ±max_weight for binary market positions.
    """
    return float(np.clip(mu_hat / (gamma * rv), -max_weight, max_weight))


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(
    gamma:      float = GAMMA,
    lookback:   int   = LOOKBACK,
    min_volume: float = 50,
    max_weight: float = MAX_WEIGHT,
    min_r2:     float = MIN_R2,
    min_price:  float = 0.0,
    limit:      int   = 50,
) -> pd.DataFrame:

    print(f"\n{Fore.CYAN}{'─'*68}")
    print(f"  Foundation Strategy  —  Return Predictability + Volatility Timing")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  γ={gamma}  lookback={lookback}d  min-vol={min_volume}  "
          f"max-weight={max_weight:.0%}  min-R²={min_r2}  limit={limit}")
    print(f"{'─'*68}{Style.RESET_ALL}\n")

    events  = fetch_events()
    markets = extract_markets(events)
    markets = [m for m in markets if float(m.get("volume_fp") or 0) >= min_volume]
    markets = [m for m in markets if not any(
        kw in (m.get("ticker") or "").lower() or kw in (m.get("title") or "").lower()
        for kw in SPORTS_KEYWORDS
    )]
    markets.sort(key=lambda m: float(m.get("volume_fp") or 0), reverse=True)
    markets = markets[:limit]
    print(f"  Markets after volume + sports filter + limit: {len(markets)}\n")

    if not markets:
        print(f"{Fore.RED}No markets. Try --min-volume 0.{Style.RESET_ALL}")
        sys.exit(0)

    rows  = []
    total = len(markets)

    for i, m in enumerate(markets):
        fill = int((i + 1) / total * 36)
        print(f"\r  [{'█'*fill}{'░'*(36-fill)}] {i+1}/{total}", end="", flush=True)

        current_p    = midpoint(m)
        if current_p is None or current_p < min_price:
            continue

        ticker       = m.get("ticker", "")
        event_ticker = m.get("event_ticker", "")
        title        = m.get("title") or ticker

        if spread_pct(m) > MAX_SPREAD:
            continue

        history = fetch_market_history(ticker, event_ticker, days=lookback + 5)
        time.sleep(0.15)

        est = estimate(history)
        if est is None or est["r2"] < min_r2:
            continue

        w = mv_weight(est["mu_hat"], est["rv"], gamma, max_weight)
        if w >= 0:  # only take BUY NO (contrarian: bet against upward trends)
            continue

        rows.append({
            "Ticker":      ticker,
            "Title":       title[:52],
            "Current P":   round(current_p, 4),
            "mu_hat":      round(est["mu_hat"], 4),   # μ̂: signal (Lecture 5.6)
            "RV":          round(est["rv"], 4),        # realized variance (Lecture 5.7)
            "Vol":         round(np.sqrt(est["rv"]), 4),
            "R2":          round(est["r2"], 3),
            "Weight":      round(w, 4),                # μ̂ / (γ·RV)
            "Action":      "BUY YES" if w > 0 else "BUY NO",
            "Signal_SR":   round(est["mu_hat"] / np.sqrt(est["rv"]), 3),  # μ̂/√RV
            "Spread_pct":  round(spread_pct(m) * 100, 2),
            "Volume":      round(float(m.get("volume_fp") or 0)),
            "Days_to_Close": days_to_close(m),
            "Hist_pts":    est["n"],
        })

    print(f"\n  Done — {len(rows)} positions passed filters.\n")

    if not rows:
        print(f"{Fore.RED}No signals passed filters. Try --min-r2 0 or --min-volume 0.{Style.RESET_ALL}")
        sys.exit(0)

    df = pd.DataFrame(rows)

    # Lecture 5.7: normalize so average |weight| = 1 (scaling constant c)
    mean_abs_w = df["Weight"].abs().mean()
    if mean_abs_w > 1e-6:
        df["Weight_scaled"] = (df["Weight"] / mean_abs_w).round(4)
    else:
        df["Weight_scaled"] = df["Weight"]

    return df.sort_values("Signal_SR", key=abs, ascending=False).reset_index(drop=True)


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, top_n: int) -> None:

    n_yes  = (df["Action"] == "BUY YES").sum()
    n_no   = (df["Action"] == "BUY NO").sum()
    avg_rv = df["RV"].mean()
    avg_sr = df["Signal_SR"].abs().mean()

    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  PORTFOLIO SUMMARY  ({len(df)} positions)")
    print(f"{'━'*68}{Style.RESET_ALL}")
    print(f"  BUY YES signals            : {n_yes}")
    print(f"  BUY NO  signals            : {n_no}")
    print(f"  Avg annualized variance    : {avg_rv:.4f}  (vol ≈ {np.sqrt(avg_rv):.1%})")
    print(f"  Avg |Signal Sharpe|        : {avg_sr:.3f}")
    print(f"  Avg |weight| (raw)         : {df['Weight'].abs().mean():.4f}")
    print(f"  Avg |weight| (scaled, c·w) : {df['Weight_scaled'].abs().mean():.4f}")

    # ── Volatility quintile analysis (Lecture 5.7) ─────────────────────────
    # Test: does the price of risk (μ̂/RV) decline as volatility rises?
    n_q = min(5, len(df))
    if n_q >= 3:
        df["Vol_Q"] = pd.qcut(df["RV"], q=n_q,
                               labels=[f"Q{i+1}" for i in range(n_q)])
        q = df.groupby("Vol_Q").agg(
            Avg_RV = ("RV",       "mean"),
            Avg_mu = ("mu_hat",   lambda x: x.abs().mean()),
            Avg_SR = ("Signal_SR",lambda x: x.abs().mean()),
            Avg_W  = ("Weight",   lambda x: x.abs().mean()),
            N      = ("Ticker",   "count"),
        )
        q["Price_of_Risk"] = q["Avg_mu"] / q["Avg_RV"]

        print(f"\n{Fore.CYAN}{'━'*68}")
        print(f"  VOLATILITY QUINTILE ANALYSIS")
        print(f"  (Lecture 5.7: price of risk should decline as RV rises)")
        print(f"{'━'*68}{Style.RESET_ALL}")
        print(f"  {'Q':<4} {'Avg RV':>8} {'Avg Vol':>8} {'|μ̂|':>8} "
              f"{'|μ̂|/RV':>9} {'Avg |SR|':>9} {'Avg |w|':>8} {'N':>4}")
        print(f"  {'─'*4} {'──────':>8} {'───────':>8} {'────':>8} "
              f"{'──────':>9} {'────────':>9} {'───────':>8} {'─':>4}")
        for label, r in q.iterrows():
            print(f"  {str(label):<4} {r['Avg_RV']:>8.4f} {np.sqrt(r['Avg_RV']):>8.1%} "
                  f"{r['Avg_mu']:>8.4f} {r['Price_of_Risk']:>9.3f} "
                  f"{r['Avg_SR']:>9.3f} {r['Avg_W']:>8.4f} {int(r['N']):>4}")

    # ── Top positions ──────────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  TOP {top_n} POSITIONS  (ranked by |Signal Sharpe| = |μ̂| / √RV)")
    print(f"{'━'*68}{Style.RESET_ALL}")

    table = []
    for _, r in df.head(top_n).iterrows():
        c = Fore.GREEN if r["Action"] == "BUY YES" else Fore.RED
        table.append([
            r["Ticker"],
            r["Title"][:34],
            f"{r['Current P']:.3f}",
            f"{r['mu_hat']:+.4f}",
            f"{r['Vol']:.1%}",
            f"{r['R2']:.2f}",
            f"{c}{r['Weight']:+.3f}{Style.RESET_ALL}",
            f"{c}{r['Weight_scaled']:+.3f}{Style.RESET_ALL}",
            f"{c}{r['Action']}{Style.RESET_ALL}",
            f"{r['Spread_pct']:.1f}%",
            r["Days_to_Close"] if r["Days_to_Close"] is not None else "?",
        ])

    headers = ["Ticker", "Title", "Curr P", "μ̂", "Vol", "R²",
               "w (raw)", "w (scaled)", "Action", "Spread", "Days"]
    print(tabulate(table, headers=headers, tablefmt="rounded_outline"))

    print(f"\n{Fore.YELLOW}Column guide:")
    print(f"  μ̂          OLS-forecasted annualized expected price change  [Lecture 5.6]")
    print(f"  Vol        Annualized realized volatility √RV               [Lecture 5.7]")
    print(f"  R²         Signal regression fit — higher = more reliable   [Lecture 5.6]")
    print(f"  w (raw)    μ̂ / (γ·RV), capped at ±{MAX_WEIGHT:.0%}                [both]")
    print(f"  w (scaled) w normalized so avg |w| = 1 (scaling constant c) [Lecture 5.7]")
    print(f"  Edges below Spread % are not exploitable after friction.")
    print(f"  Not financial advice.{Style.RESET_ALL}\n")


# ── Walk-forward backtest ──────────────────────────────────────────────────────

def simulate_market(
    ticker:      str,
    title:       str,
    prices:      list[float],
    sprd:        float,
    lookback:    int,
    hold_period: int,
    gamma:       float,
    max_weight:  float,
    min_r2:      float,
    min_price:   float = 0.0,
) -> list[dict]:
    """
    Walk forward through a single market's price history.

    At each step t (after the lookback burn-in):
      1. Estimate μ̂ and RV from prices[t-lookback : t]      [Lectures 5.6 + 5.7]
      2. Compute weight w = μ̂ / (γ · RV)                   [both combined]
      3. Record P&L over the next hold_period days

    P&L per unit of bankroll:
        raw_pnl  = w · (exit_price − entry_price)
        net_pnl  = raw_pnl − spread_cost
    """
    trades = []
    n = len(prices)

    for t in range(lookback, n - hold_period):
        hist = prices[t - lookback : t]
        est  = estimate(hist)
        if est is None or est["r2"] < min_r2:
            continue

        w = mv_weight(est["mu_hat"], est["rv"], gamma, max_weight)
        if w >= 0:  # only take BUY NO (contrarian: bet against upward trends)
            continue

        entry = prices[t]
        if entry < min_price:
            continue
        exit_ = prices[t + hold_period]

        # P&L = w · ΔP  (positive when price moves in forecasted direction)
        raw_pnl     = w * (exit_ - entry)
        spread_cost = abs(w) * sprd * entry / 2   # half-spread paid at entry

        # Kalshi fee: 0.07 × C × P × (1-P)
        # For BUY NO: C = |w| / (1-entry),  P = entry (YES price)
        # Substituting: 0.07 × [|w|/(1-entry)] × entry × (1-entry) = 0.07 × |w| × entry
        kalshi_fee  = 0.07 * abs(w) * entry

        net_pnl     = raw_pnl - spread_cost - kalshi_fee

        trades.append({
            "Ticker":     ticker,
            "Title":      title[:45],
            "Day":        t,
            "Entry P":    round(entry, 4),
            "Exit P":     round(exit_, 4),
            "Weight":     round(w, 4),
            "Action":     "BUY YES" if w > 0 else "BUY NO",
            "mu_hat":     round(est["mu_hat"], 4),
            "RV":         round(est["rv"], 4),
            "R2":         round(est["r2"], 3),
            "Raw PnL":    round(raw_pnl, 6),
            "Net PnL":    round(net_pnl, 6),
            "Spread":     round(sprd, 4),
            "Kalshi Fee": round(kalshi_fee, 6),
            "Win":        net_pnl > 0,
        })

    return trades


def run_backtest(
    gamma:       float = GAMMA,
    lookback:    int   = LOOKBACK,
    hold_period: int   = 1,
    min_volume:  float = 50,
    max_weight:  float = MAX_WEIGHT,
    min_r2:      float = MIN_R2,
    min_price:   float = 0.0,
    fetch_days:  int   = 90,
    limit:       int   = 50,
) -> pd.DataFrame:

    print(f"\n{Fore.CYAN}{'─'*68}")
    print(f"  Foundation Strategy — Walk-Forward Backtest")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  γ={gamma}  lookback={lookback}d  hold={hold_period}d  "
          f"fetch={fetch_days}d  min-R²={min_r2}  min-price={min_price}  limit={limit}")
    print(f"{'─'*68}{Style.RESET_ALL}\n")

    events  = fetch_events()
    markets = extract_markets(events)
    markets = [m for m in markets if float(m.get("volume_fp") or 0) >= min_volume]
    markets = [m for m in markets if not any(
        kw in (m.get("ticker") or "").lower() or kw in (m.get("title") or "").lower()
        for kw in SPORTS_KEYWORDS
    )]
    markets.sort(key=lambda m: float(m.get("volume_fp") or 0), reverse=True)
    markets = markets[:limit]
    print(f"  Markets after volume + sports filter + limit: {len(markets)}\n")
    print(f"  Fetching {fetch_days}-day history and simulating…\n")

    all_trades = []
    total      = len(markets)

    for i, m in enumerate(markets):
        fill = int((i + 1) / total * 36)
        print(f"\r  [{'█'*fill}{'░'*(36-fill)}] {i+1}/{total}  "
              f"({len(all_trades)} trades)", end="", flush=True)

        ticker       = m.get("ticker", "")
        event_ticker = m.get("event_ticker", "")
        title        = m.get("title") or ticker
        sprd         = spread_pct(m)

        if sprd > MAX_SPREAD:
            continue

        prices = fetch_market_history(ticker, event_ticker, days=fetch_days)
        time.sleep(0.15)

        if len(prices) < lookback + hold_period + 1:
            continue

        trades = simulate_market(
            ticker=ticker, title=title, prices=prices, sprd=sprd,
            lookback=lookback, hold_period=hold_period,
            gamma=gamma, max_weight=max_weight, min_r2=min_r2,
            min_price=min_price,
        )
        all_trades.extend(trades)

    print(f"\n  Simulation complete — {len(all_trades)} trades.\n")

    if not all_trades:
        print(f"{Fore.RED}No trades generated. Try --min-r2 0 or --min-volume 0.{Style.RESET_ALL}")
        sys.exit(0)

    return pd.DataFrame(all_trades)


def _sub_stats(sub: pd.DataFrame, ann_factor: float) -> tuple:
    """(n, win_pct, mean_net, sharpe, t_stat, p_val) for a trades sub-sample."""
    net = sub["Net PnL"]
    n   = len(net)
    if n < 2:
        return n, 0.0, float("nan"), float("nan"), 0.0, 1.0
    mn   = net.mean()
    sd   = net.std(ddof=1)
    sr   = (mn / sd) * ann_factor if sd > 1e-8 else 0.0
    t, p = stats.ttest_1samp(net, 0)
    win  = sub["Win"].mean() * 100
    return n, win, mn, sr, t, p


def print_backtest_report(df: pd.DataFrame, hold_period: int, top_n: int) -> None:
    n_trades  = len(df)
    n_markets = df["Ticker"].nunique()
    win_rate  = df["Win"].mean() * 100

    raw = df["Raw PnL"]
    net = df["Net PnL"]

    mean_raw = raw.mean()
    mean_net = net.mean()
    std_net  = net.std(ddof=1)

    ann_factor   = np.sqrt(252 / hold_period)
    sharpe_raw   = (raw.mean() / raw.std(ddof=1)) * ann_factor if raw.std(ddof=1) > 1e-8 else 0.0
    sharpe_net   = (mean_net / std_net)            * ann_factor if std_net          > 1e-8 else 0.0

    cumulative   = net.cumsum()
    max_dd       = (cumulative - cumulative.cummax()).min()

    gross_win    = net[net > 0].sum()
    gross_loss   = net[net < 0].abs().sum()
    profit_factor = gross_win / gross_loss if gross_loss > 1e-8 else float("inf")

    sr_color = Fore.GREEN if sharpe_net > 0.5 else (Fore.YELLOW if sharpe_net > 0 else Fore.RED)

    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  BACKTEST RESULTS  ({n_trades} trades across {n_markets} markets)")
    print(f"{'━'*68}{Style.RESET_ALL}")
    print(f"  Hold period              : {hold_period} day(s)")
    print(f"  Win rate                 : {win_rate:.1f}%  ({df['Win'].sum()}/{n_trades})")
    print(f"  Mean raw P&L / trade     : {mean_raw:+.6f}")
    print(f"  Mean net P&L / trade     : {mean_net:+.6f}  (after spread cost)")
    print(f"  Profit factor            : {profit_factor:.2f}x")
    print(f"  Max drawdown (cumul.)    : {max_dd:+.6f}")
    print(f"  Annualized Sharpe (raw)  : {sr_color}{sharpe_raw:+.3f}{Style.RESET_ALL}")
    print(f"  Annualized Sharpe (net)  : {sr_color}{sharpe_net:+.3f}{Style.RESET_ALL}")

    # ── Empirical results ─────────────────────────────────────────────────
    t_stat_mean, p_val_mean = stats.ttest_1samp(net, 0)
    ci_lo, ci_hi = stats.t.interval(
        0.95, df=n_trades - 1,
        loc=mean_net, scale=stats.sem(net)
    )
    rng = np.random.default_rng(42)
    bs_sharpes = []
    for _ in range(2000):
        samp = rng.choice(net.values, size=n_trades, replace=True)
        sd   = samp.std(ddof=1)
        if sd > 1e-8:
            bs_sharpes.append((samp.mean() / sd) * ann_factor)
    sh_lo, sh_hi = np.percentile(bs_sharpes, [2.5, 97.5])

    sig_mean = ("***" if p_val_mean < 0.01 else
                "**"  if p_val_mean < 0.05 else
                "*"   if p_val_mean < 0.10 else "ns")
    pct_net = (mean_net / mean_raw * 100) if abs(mean_raw) > 1e-8 else float("nan")

    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  EMPIRICAL RESULTS")
    print(f"{'━'*68}{Style.RESET_ALL}")
    print(f"  t-stat (H₀: μ_net=0)      : {t_stat_mean:+.3f}  (p={p_val_mean:.4f} {sig_mean})")
    print(f"  95% CI for mean net PnL   : [{ci_lo:+.6f},  {ci_hi:+.6f}]")
    print(f"  Bootstrap 95% CI Sharpe   : [{sh_lo:+.3f},  {sh_hi:+.3f}]  (2,000 resamples)")
    mean_fee    = df["Kalshi Fee"].mean()
    mean_spread = df["Net PnL"].add(df["Kalshi Fee"]).rsub(mean_raw).mean()  # raw - fee_adj - net ≈ spread
    mean_spread = (raw - net - df["Kalshi Fee"]).mean()

    print(f"  Spread drag               : {mean_spread:+.6f} per trade")
    print(f"  Kalshi fee drag           : {mean_fee:+.6f} per trade  "
          f"(0.07 × |w| × entry)")
    print(f"  Total cost drag           : {mean_raw - mean_net:+.6f} per trade  "
          f"({100 - pct_net:.1f}% of gross PnL)")

    # ── Vol timing check (Lecture 5.7): does high RV → lower realized P&L? ──
    n_q = min(5, n_trades)
    if n_q >= 3:
        df["RV_Q"] = pd.qcut(df["RV"], q=min(5, n_trades), labels=[f"Q{i+1}" for i in range(min(5, n_trades))])
        q = df.groupby("RV_Q").agg(
            Avg_RV     = ("RV",      "mean"),
            Avg_W      = ("Weight",  lambda x: x.abs().mean()),
            Mean_Net   = ("Net PnL", "mean"),
            Win_Rate   = ("Win",     lambda x: x.mean() * 100),
            N          = ("Ticker",  "count"),
        )
        print(f"\n{Fore.CYAN}{'━'*68}")
        print(f"  VOL QUINTILE P&L  (Lecture 5.7: lower vol → better risk-adjusted returns?)")
        print(f"{'━'*68}{Style.RESET_ALL}")
        print(f"  {'Q':<4} {'Avg RV':>8} {'Avg Vol':>8} {'Avg |w|':>8} {'Mean Net PnL':>13} {'Win %':>7} {'N':>5}")
        print(f"  {'─'*4} {'──────':>8} {'───────':>8} {'───────':>8} {'────────────':>13} {'─────':>7} {'─':>5}")
        for label, r in q.iterrows():
            c = Fore.GREEN if r["Mean_Net"] > 0 else Fore.RED
            print(f"  {str(label):<4} {r['Avg_RV']:>8.4f} {np.sqrt(r['Avg_RV']):>8.1%} "
                  f"{r['Avg_W']:>8.4f} "
                  f"{c}{r['Mean_Net']:>+13.6f}{Style.RESET_ALL} "
                  f"{r['Win_Rate']:>7.1f}% {int(r['N']):>5}")

    # ── P&L distribution ──────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  NET P&L DISTRIBUTION")
    print(f"{'━'*68}{Style.RESET_ALL}")
    for pct, val in zip([5, 25, 50, 75, 95], np.percentile(net, [5, 25, 50, 75, 95])):
        bar = "█" * min(int(abs(val) * 2000), 30)
        c   = Fore.GREEN if val >= 0 else Fore.RED
        print(f"  p{pct:<3}  {c}{val:+.6f}{Style.RESET_ALL}  {bar}")

    # ── Direction breakdown ────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  BY DIRECTION")
    print(f"{'━'*68}{Style.RESET_ALL}")
    for label, sub in [("BUY YES", df[df["Action"] == "BUY YES"]),
                        ("BUY NO",  df[df["Action"] == "BUY NO"])]:
        if sub.empty:
            continue
        c = Fore.GREEN if label == "BUY YES" else Fore.RED
        sr = (sub["Net PnL"].mean() / sub["Net PnL"].std(ddof=1)) * ann_factor \
             if sub["Net PnL"].std(ddof=1) > 1e-8 else 0.0
        print(f"  {c}{label}{Style.RESET_ALL}  n={len(sub):>5}  "
              f"win={sub['Win'].mean()*100:.1f}%  "
              f"mean net={sub['Net PnL'].mean():+.6f}  "
              f"Sharpe={sr:+.3f}")

    # ── Sub-sample analysis ────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  SUB-SAMPLE ANALYSIS")
    print(f"{'━'*68}{Style.RESET_ALL}")

    def _fmt_sub(label, n, win, mn, sr, t, p):
        sig = ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns")
        c   = Fore.GREEN if (isinstance(mn, float) and mn > 0) else Fore.RED
        mn_str = f"{mn:>+.5f}" if isinstance(mn, float) and not np.isnan(mn) else "   n/a  "
        sr_str = f"{sr:>+.3f}" if isinstance(sr, float) and not np.isnan(sr) else "  n/a"
        return (f"  {label:<32}  n={n:>5}  win={win:>5.1f}%  "
                f"net={c}{mn_str}{Style.RESET_ALL}  "
                f"SR={sr_str}  t={t:>+.2f} ({sig})")

    mid = len(df) // 2
    r2m = df["R2"].median()

    print(f"  Temporal split (first vs. second half of trades)")
    for label, sub in [("First half  (trades 1–N/2)", df.iloc[:mid]),
                        ("Second half (trades N/2–N)", df.iloc[mid:])]:
        print(_fmt_sub(label, *_sub_stats(sub, ann_factor)))

    print(f"\n  Signal confidence split (R² median = {r2m:.3f})")
    for label, sub in [("Low R²  (< median)", df[df["R2"] <  r2m]),
                        ("High R² (≥ median)", df[df["R2"] >= r2m])]:
        print(_fmt_sub(label, *_sub_stats(sub, ann_factor)))

    p40, p60 = df["Entry P"].quantile(0.40), df["Entry P"].quantile(0.60)
    print(f"\n  Entry price split (P<{p40:.2f} / {p40:.2f}≤P<{p60:.2f} / P≥{p60:.2f})")
    for label, sub in [
        (f"Low   entry (P < {p40:.2f})",              df[df["Entry P"] <  p40]),
        (f"Mid   entry ({p40:.2f} ≤ P < {p60:.2f})", df[(df["Entry P"] >= p40) & (df["Entry P"] < p60)]),
        (f"High  entry (P ≥ {p60:.2f})",              df[df["Entry P"] >= p60]),
    ]:
        print(_fmt_sub(label, *_sub_stats(sub, ann_factor)))

    # ── Robustness checks ─────────────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  ROBUSTNESS CHECKS")
    print(f"{'━'*68}{Style.RESET_ALL}")

    print(f"  Sensitivity to min-R² threshold (post-hoc filter on backtest trades)")
    print(f"  {'min-R²':>7}  {'N':>6}  {'Win%':>6}  {'Mean Net':>11}  {'Sharpe':>7}  p-val")
    print(f"  {'──────':>7}  {'─':>6}  {'────':>6}  {'────────':>11}  {'──────':>7}  ─────")
    for r2_thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        sub = df[df["R2"] >= r2_thresh]
        if len(sub) < 2:
            print(f"  {r2_thresh:>7.2f}  {len(sub):>6}  — (too few trades)")
            continue
        n_s, win_s, mn_s, sr_s, _, p_s = _sub_stats(sub, ann_factor)
        sig = ("***" if p_s < 0.01 else "**" if p_s < 0.05 else "*" if p_s < 0.10 else "ns")
        c   = Fore.GREEN if mn_s > 0 else Fore.RED
        print(f"  {r2_thresh:>7.2f}  {n_s:>6}  {win_s:>5.1f}%  "
              f"{c}{mn_s:>+11.6f}{Style.RESET_ALL}  {sr_s:>+7.3f}  {p_s:.4f} {sig}")

    print(f"\n  Gross vs. net PnL (spread-cost sensitivity)")
    for label, series in [("Gross — no spread cost", raw), ("Net   — with spread cost", net)]:
        n_s  = len(series)
        mn_s = series.mean()
        sd_s = series.std(ddof=1)
        sr_s = (mn_s / sd_s) * ann_factor if sd_s > 1e-8 else 0.0
        _, p_s = stats.ttest_1samp(series, 0)
        sig = ("***" if p_s < 0.01 else "**" if p_s < 0.05 else "*" if p_s < 0.10 else "ns")
        c   = Fore.GREEN if mn_s > 0 else Fore.RED
        print(f"  {label:<26}  n={n_s:>5}  "
              f"mean={c}{mn_s:>+.6f}{Style.RESET_ALL}  "
              f"SR={sr_s:>+.3f}  p={p_s:.4f} {sig}")

    top_mkt      = df.groupby("Ticker")["Net PnL"].count().idxmax()
    top_mkt_n    = (df["Ticker"] == top_mkt).sum()
    sub_loo      = df[df["Ticker"] != top_mkt]
    print(f"\n  Leave-one-market-out (drop {top_mkt}, {top_mkt_n} trades)")
    if len(sub_loo) >= 2:
        n_s, win_s, mn_s, sr_s, _, p_s = _sub_stats(sub_loo, ann_factor)
        sig = ("***" if p_s < 0.01 else "**" if p_s < 0.05 else "*" if p_s < 0.10 else "ns")
        c   = Fore.GREEN if mn_s > 0 else Fore.RED
        print(f"  Without {top_mkt:<22}  n={n_s:>5}  win={win_s:.1f}%  "
              f"mean={c}{mn_s:>+.6f}{Style.RESET_ALL}  SR={sr_s:>+.3f}  p={p_s:.4f} {sig}")

    # ── Top markets by trade count ─────────────────────────────────────────
    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  TOP {top_n} MARKETS BY TRADE COUNT")
    print(f"{'━'*68}{Style.RESET_ALL}")
    mkt = (
        df.groupby(["Ticker", "Title"])
        .agg(Trades=("Win","count"), Win_Rate=("Win", lambda x: x.mean()*100),
             Mean_Net=("Net PnL","mean"), Total_Net=("Net PnL","sum"))
        .sort_values("Trades", ascending=False)
        .head(top_n).reset_index()
    )
    table = []
    for _, r in mkt.iterrows():
        c = Fore.GREEN if r["Total_Net"] > 0 else Fore.RED
        table.append([r["Ticker"], r["Title"][:36], int(r["Trades"]),
                      f"{r['Win_Rate']:.1f}%", f"{r['Mean_Net']:+.6f}",
                      f"{c}{r['Total_Net']:+.6f}{Style.RESET_ALL}"])
    print(tabulate(table,
                   headers=["Ticker","Title","Trades","Win %","Mean Net","Total Net"],
                   tablefmt="rounded_outline"))

    print(f"\n{Fore.YELLOW}⚠  Survivorship bias: only currently-open markets are tested.")
    print(f"   Spread cost = half-spread × |w| × entry price, deducted at entry.")
    print(f"   Annualized Sharpe assumes {int(252/hold_period)} non-overlapping periods/year.")
    print(f"   Not financial advice.{Style.RESET_ALL}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Foundation Strategy: Return Predictability + Volatility Timing"
    )
    p.add_argument("--gamma",      type=float, default=GAMMA,
                   help=f"Risk aversion γ (default {GAMMA})")
    p.add_argument("--lookback",   type=int,   default=LOOKBACK,
                   help=f"Lookback window in days (default {LOOKBACK})")
    p.add_argument("--hold",       type=int,   default=7,
                   help="Hold period in days for backtest (default 7)")
    p.add_argument("--min-volume", type=float, default=50,
                   help="Min market volume (default 50)")
    p.add_argument("--max-weight", type=float, default=MAX_WEIGHT,
                   help=f"Max raw position size (default {MAX_WEIGHT})")
    p.add_argument("--min-r2",     type=float, default=MIN_R2,
                   help=f"Min R² to trust signal (default {MIN_R2})")
    p.add_argument("--fetch-days", type=int,   default=180,
                   help="Days of history to fetch per market in backtest (default 180)")
    p.add_argument("--limit",      type=int,   default=200,
                   help="Max markets to process, ranked by volume (default 50)")
    p.add_argument("--top-n",      type=int,   default=20,
                   help="Positions to display (default 20)")
    p.add_argument("--min-price",  type=float, default=0.0,
                   help="Skip trades where entry price (YES prob) is below this (default 0.0)")
    p.add_argument("--export",     metavar="FILE.csv",
                   help="Save full results to CSV")
    p.add_argument("--backtest",   action="store_true",
                   help="Run walk-forward backtest instead of live screener")
    args = p.parse_args()

    if args.backtest:
        df = run_backtest(
            gamma=args.gamma,
            lookback=args.lookback,
            hold_period=args.hold,
            min_volume=args.min_volume,
            max_weight=args.max_weight,
            min_r2=args.min_r2,
            min_price=args.min_price,
            fetch_days=args.fetch_days,
            limit=args.limit,
        )
        print_backtest_report(df, hold_period=args.hold, top_n=args.top_n)
    else:
        df = run(
            gamma=args.gamma,
            lookback=args.lookback,
            min_volume=args.min_volume,
            max_weight=args.max_weight,
            min_r2=args.min_r2,
            min_price=args.min_price,
            limit=args.limit,
        )
        print_report(df, top_n=args.top_n)

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"{Fore.GREEN}✓ Saved {len(df)} rows → {args.export}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
