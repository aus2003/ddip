"""
Foundation Strategy — Kalshi Prediction Markets (MEAN REVERSION V6 - 1-DAY HOLD)
================================================
Applies two techniques from finance lecture notes to prediction markets:
1. Return Predictability (Lecture 5.6):
   A signal forecasts expected returns. Here: OLS regression slope over
   a lookback window estimates the expected daily price change (μ̂).
   (Negated to fade trends — betting on mean reversion).
2. Volatility Timing (Lecture 5.7):
   Scale positions by inverse realized variance (1/RV). 
Combined mean-variance optimal weight:
    w = μ̂ / (γ · RV)
DESIGN CHOICES
==============
1-12. [Mean-Reversion and All-Markets logic applied]
13. 1-Day Hold / Mean Reversion Only (MERGED)
    This version strictly trades negative momentum/mean-reversion signals (μ̂ < 0). 
    Positive signals are discarded. When a valid signal fires, the backtester enters 
    a BUY NO position on day `t` and closes it exactly one day later on day `t+1`.
Usage:
    python model.py                                # live screener 
    python model.py --backtest                     # backtest (1-Day Hold)
    python model.py --backtest --min-price 0.15    # filter low-price trades
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
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False
colorama_init(autoreset=True)
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS  = {"Accept": "application/json"}

# ── Defaults ──────────────────────────────────────────────────────────────────
GAMMA       = 2.0   
LOOKBACK    = 30    
MIN_R2      = 0.10  
MAX_WEIGHT  = 0.25  
MIN_HISTORY = 10    
MAX_SPREAD  = 0.10  

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
    print(f"{Fore.YELLOW}Fetching all OPEN events…{Style.RESET_ALL}", flush=True)
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
    try:
        series_ticker = event_ticker.split("-")[0] if event_ticker else ""
        if not series_ticker:
            return []
        
        end_ts = int(time.time())
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
    n = len(prices)
    if n < MIN_HISTORY:
        return None
    changes = np.diff(np.array(prices, dtype=float))  
    if len(changes) < 2:
        return None
    x = np.arange(len(changes), dtype=float)
    slope, _, r_val, _, _ = stats.linregress(x, changes)
    r2 = r_val ** 2
    rv = float(np.var(changes, ddof=1)) * 252   
    if rv < 1e-8:
        return None
    return {
        "mu_hat": -slope * 252,  # Negative momentum / Fading the trend
        "rv":     rv,            
        "r2":     r2,
        "n":      n,
    }

def mv_weight(mu_hat: float, rv: float, gamma: float, max_weight: float) -> float:
    return float(np.clip(mu_hat / (gamma * rv), -max_weight, max_weight))

# ── Main pipeline (Live Screener) ──────────────────────────────────────────────
def run(
    gamma:      float = GAMMA,
    lookback:   int   = LOOKBACK,
    min_volume: float = 50,
    max_weight: float = MAX_WEIGHT,
    min_r2:     float = MIN_R2,
    min_price:  float = 0.15,
    limit:      int   = 50,
) -> pd.DataFrame:
    print(f"\n{Fore.CYAN}{'─'*68}")
    print(f"  Foundation Strategy  —  MEAN REVERSION (1-Day Hold, BUY NO Only)")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  γ={gamma}  lookback={lookback}d  min-vol={min_volume}  "
          f"max-weight={max_weight:.0%}  min-R²={min_r2}  limit={limit}")
    print(f"{'─'*68}{Style.RESET_ALL}\n")
    events  = fetch_events()
    markets = extract_markets(events)
    markets = [m for m in markets if float(m.get("volume_fp") or 0) >= min_volume]
    markets.sort(key=lambda m: float(m.get("volume_fp") or 0), reverse=True)
    markets = markets[:limit]
    print(f"  Markets after volume filter + limit: {len(markets)}\n")
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
        
        # Mean Reversion logic: strictly filter for BUY NO
        if w >= 0:
            continue
            
        rows.append({
            "Ticker":      ticker,
            "Title":       title[:52],
            "Current P":   round(current_p, 4),
            "mu_hat":      round(est["mu_hat"], 4),   
            "RV":          round(est["rv"], 4),        
            "Vol":         round(np.sqrt(est["rv"]), 4),
            "R2":          round(est["r2"], 3),
            "Weight":      round(w, 4),                
            "Action":      "BUY NO", 
            "Signal_SR":   round(est["mu_hat"] / np.sqrt(est["rv"]), 3),  
            "Spread_pct":  round(spread_pct(m) * 100, 2),
            "Volume":      round(float(m.get("volume_fp") or 0)),
            "Days_to_Close": days_to_close(m),
            "Hist_pts":    est["n"],
        })
    print(f"\n  Done — {len(rows)} signals passed filters.\n")
    if not rows:
        print(f"{Fore.RED}No signals passed filters. Try --min-r2 0 or --min-volume 0.{Style.RESET_ALL}")
        sys.exit(0)
    df = pd.DataFrame(rows)
    mean_abs_w = df["Weight"].abs().mean()
    if mean_abs_w > 1e-6:
        df["Weight_scaled"] = (df["Weight"] / mean_abs_w).round(4)
    else:
        df["Weight_scaled"] = df["Weight"]
    return df.sort_values("Signal_SR", key=abs, ascending=False).reset_index(drop=True)

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
    table = []
    for _, r in df.head(top_n).iterrows():
        c = Fore.GREEN if r["Action"] == "BUY YES" else Fore.RED
        table.append([
            r["Ticker"], r["Title"][:34], f"{r['Current P']:.3f}",
            f"{r['mu_hat']:+.4f}", f"{r['Vol']:.1%}", f"{r['R2']:.2f}",
            f"{c}{r['Weight']:+.3f}{Style.RESET_ALL}", f"{c}{r['Action']}{Style.RESET_ALL}",
            f"{r['Spread_pct']:.1f}%"
        ])
    headers = ["Ticker", "Title", "Curr P", "μ̂", "Vol", "R²", "w (raw)", "Action", "Spread"]
    print("\n" + tabulate(table, headers=headers, tablefmt="rounded_outline"))

# ── Walk-forward backtest ──────────────────────────────────────────────────────
def simulate_market(
    ticker:      str,
    title:       str,
    prices:      list[float],
    sprd:        float,
    lookback:    int,
    gamma:       float,
    max_weight:  float,
    min_r2:      float,
    min_price:   float = 0.15,
) -> list[dict]:
    trades = []
    n = len(prices)
    if n <= lookback:
        return trades
    for t in range(lookback, n - 1):
        hist = prices[t - lookback : t]
        est  = estimate(hist)
        if est is None or est["r2"] < min_r2:
            continue
        w = mv_weight(est["mu_hat"], est["rv"], gamma, max_weight)
        
        # Mean Reversion logic: Ignore positive signals
        if w >= 0:
            continue
        
        entry = prices[t]
        if entry < min_price:
            continue
            
        exit_ = prices[t + 1]
        hold_days = 1
        raw_pnl     = w * (exit_ - entry)
        
        spread_cost = (abs(w) * sprd * entry / 2) + (abs(w) * sprd * exit_ / 2)
        kalshi_fee  = 0.07 * abs(w) * entry 
        
        no_fee_pnl  = raw_pnl - spread_cost
        net_pnl     = no_fee_pnl - kalshi_fee
        
        trades.append({
            "Ticker":      ticker,
            "Title":       title[:45],
            "Day":         t,
            "Entry P":     round(entry, 4),
            "Exit P":      round(exit_, 4),
            "Hold Days":   hold_days,
            "Weight":      round(w, 4),
            "Action":      "BUY NO", 
            "mu_hat":      round(est["mu_hat"], 4),
            "RV":          round(est["rv"], 4),
            "R2":          round(est["r2"], 3),
            "Raw PnL":     round(raw_pnl, 6),
            "Spread Cost": round(spread_cost, 6),
            "Kalshi Fee":  round(kalshi_fee, 6),
            "No Fee PnL":  round(no_fee_pnl, 6),
            "Net PnL":     round(net_pnl, 6),
            "Spread":      round(sprd, 4),
            "Win (Raw)":   raw_pnl > 0,
            "Win (NoFee)": no_fee_pnl > 0,
            "Win (Net)":   net_pnl > 0,
        })
    return trades

def run_backtest(
    gamma:       float = GAMMA,
    lookback:    int   = LOOKBACK,
    min_volume:  float = 50,
    max_weight:  float = MAX_WEIGHT,
    min_r2:      float = MIN_R2,
    min_price:   float = 0.15,
    fetch_days:  int   = 90,
    limit:       int   = 50,
) -> pd.DataFrame:
    print(f"\n{Fore.CYAN}{'─'*68}")
    print(f"  Foundation Strategy — Walk-Forward Backtest (1-Day Hold)")
    print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  γ={gamma}  lookback={lookback}d  fetch={fetch_days}d  "
          f"min-R²={min_r2}  min-price={min_price}  limit={limit}")
    print(f"{'─'*68}{Style.RESET_ALL}\n")
    events  = fetch_events()
    markets = extract_markets(events)
    markets = [m for m in markets if float(m.get("volume_fp") or 0) >= min_volume]
    markets.sort(key=lambda m: float(m.get("volume_fp") or 0), reverse=True)
    markets = markets[:limit]
    print(f"  Markets after volume filter + limit: {len(markets)}\n")
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
        trades = simulate_market(
            ticker=ticker, title=title, prices=prices, sprd=sprd,
            lookback=lookback, gamma=gamma, max_weight=max_weight, 
            min_r2=min_r2, min_price=min_price,
        )
        all_trades.extend(trades)
    print(f"\n  Simulation complete — {len(all_trades)} trades.\n")
    if not all_trades:
        print(f"{Fore.RED}No trades generated. Try --min-r2 0 or --min-volume 0.{Style.RESET_ALL}")
        sys.exit(0)
    return pd.DataFrame(all_trades)

def _sr_se(net: pd.Series) -> float:
    sr = net.mean() / net.std(ddof=1)
    T  = len(net)
    return float(np.sqrt((1 / (T - 1)) * (1 + sr**2 / 2)))

def _plot_backtest(df: pd.DataFrame, prefix: str = "backtest") -> None:
    if not _MATPLOTLIB:
        print(f"{Fore.YELLOW}matplotlib not installed — skipping plots.{Style.RESET_ALL}")
        return
    net = df["Net PnL"].reset_index(drop=True)
    cum = (1 + net).cumprod()
    dd  = (cum - cum.cummax()) / cum.cummax()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    cum.plot(ax=ax1, linewidth=1.5, color="steelblue")
    ax1.set_title("Foundation Strategy — Cumulative Net Performance")
    ax1.set_ylabel("Growth of $1")
    ax1.grid(True, alpha=0.3)
    dd.plot(ax=ax2, linewidth=1, color="firebrick")
    ax2.fill_between(dd.index, dd, alpha=0.2, color="firebrick")
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{prefix}_perf.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"{Fore.GREEN}✓ Saved {path}{Style.RESET_ALL}")

def print_backtest_report(df: pd.DataFrame, top_n: int, plot: bool = False) -> None:
    n_trades  = len(df)
    n_markets = df["Ticker"].nunique()
    
    # ── P&L Series ────────────────────────────────────────────────────────
    raw = df["Raw PnL"]
    no_fee = df["No Fee PnL"]
    net = df["Net PnL"]
    
    avg_hold_days = df["Hold Days"].mean()
    ann_factor    = np.sqrt(252 / avg_hold_days) if avg_hold_days > 0 else 0.0
    
    def get_emp(series):
        if len(series) < 2: return 0.0, 1.0, 0.0, 0.0
        t, p = stats.ttest_1samp(series, 0)
        ci_lo, ci_hi = stats.t.interval(0.95, df=len(series) - 1, loc=series.mean(), scale=stats.sem(series))
        return t, p, ci_lo, ci_hi

    def format_sig(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "ns"

    t_raw, p_raw, ci_lo_raw, ci_hi_raw = get_emp(raw)
    t_nofee, p_nofee, ci_lo_nofee, ci_hi_nofee = get_emp(no_fee)
    t_net, p_net, ci_lo_net, ci_hi_net = get_emp(net)

    # Raw stats
    win_rate_raw  = df["Win (Raw)"].mean() * 100
    mean_raw      = raw.mean()
    sharpe_raw    = (mean_raw / raw.std(ddof=1)) * ann_factor if raw.std(ddof=1) > 1e-8 else 0.0
    
    # No-Fee stats (Spread Only)
    win_rate_nofee = df["Win (NoFee)"].mean() * 100
    mean_nofee     = no_fee.mean()
    sharpe_nofee   = (mean_nofee / no_fee.std(ddof=1)) * ann_factor if no_fee.std(ddof=1) > 1e-8 else 0.0
    
    # Net stats (Spread + Kalshi Fee)
    win_rate_net  = df["Win (Net)"].mean() * 100
    mean_net      = net.mean()
    std_net       = net.std(ddof=1)
    sharpe_net    = (mean_net / std_net) * ann_factor if std_net > 1e-8 else 0.0
    
    cumulative   = net.cumsum()
    max_dd       = (cumulative - cumulative.cummax()).min()
    
    def get_color(val):
        return Fore.GREEN if val > 0 else (Fore.RED if val < 0 else Fore.YELLOW)
    
    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  BACKTEST RESULTS  ({n_trades} trades across {n_markets} markets)")
    print(f"{'━'*68}{Style.RESET_ALL}")
    print(f"  Avg Hold period          : {avg_hold_days:.1f} day(s)")
    print(f"  Max drawdown (net cumul.): {max_dd:+.6f}\n")
    
    print(f"  {Fore.YELLOW}[1] RAW MARKET (0 Friction){Style.RESET_ALL}")
    print(f"      Win Rate             : {win_rate_raw:.1f}%")
    print(f"      Mean P&L / trade     : {get_color(mean_raw)}{mean_raw:+.6f}{Style.RESET_ALL}")
    print(f"      Annualized Sharpe    : {get_color(sharpe_raw)}{sharpe_raw:+.3f}{Style.RESET_ALL}")
    print(f"      t-stat (H₀: μ=0)     : {t_raw:+.3f}  (p={p_raw:.4f} {format_sig(p_raw)})")
    print(f"      95% CI for mean P&L  : [{ci_lo_raw:+.6f},  {ci_hi_raw:+.6f}]\n")

    print(f"  {Fore.YELLOW}[2] EXCLUDING KALSHI FEES (Spread Drag Only){Style.RESET_ALL}")
    print(f"      Win Rate             : {win_rate_nofee:.1f}%")
    print(f"      Mean P&L / trade     : {get_color(mean_nofee)}{mean_nofee:+.6f}{Style.RESET_ALL}")
    print(f"      Annualized Sharpe    : {get_color(sharpe_nofee)}{sharpe_nofee:+.3f}{Style.RESET_ALL}")
    print(f"      t-stat (H₀: μ=0)     : {t_nofee:+.3f}  (p={p_nofee:.4f} {format_sig(p_nofee)})")
    print(f"      95% CI for mean P&L  : [{ci_lo_nofee:+.6f},  {ci_hi_nofee:+.6f}]\n")
    
    print(f"  {Fore.YELLOW}[3] FULLY NET (Spread + Kalshi Fees){Style.RESET_ALL}")
    print(f"      Win Rate             : {win_rate_net:.1f}%")
    print(f"      Mean P&L / trade     : {get_color(mean_net)}{mean_net:+.6f}{Style.RESET_ALL}")
    print(f"      Annualized Sharpe    : {get_color(sharpe_net)}{sharpe_net:+.3f}{Style.RESET_ALL}")
    print(f"      t-stat (H₀: μ=0)     : {t_net:+.3f}  (p={p_net:.4f} {format_sig(p_net)})")
    print(f"      95% CI for mean P&L  : [{ci_lo_net:+.6f},  {ci_hi_net:+.6f}]\n")
    
    mean_spread_cost = df["Spread Cost"].mean()
    mean_kalshi_fee  = df["Kalshi Fee"].mean()
    total_cost       = mean_spread_cost + mean_kalshi_fee
    
    print(f"{Fore.CYAN}{'━'*68}")
    print(f"  TRANSACTION COSTS (Averages)")
    print(f"{'━'*68}{Style.RESET_ALL}")
    print(f"  Spread Cost (Entry + Exit)    : {mean_spread_cost:+.6f} per trade")
    print(f"  Kalshi Fee (Entry Only)       : {mean_kalshi_fee:+.6f} per trade")
    print(f"  Total Cost                    : {total_cost:+.6f} per trade")
    
    print(f"\n{Fore.CYAN}{'━'*68}")
    print(f"  TOP {top_n} MARKETS BY TRADE COUNT")
    print(f"{'━'*68}{Style.RESET_ALL}")
    mkt = (
        df.groupby(["Ticker", "Title"])
        .agg(Trades=("Win (Net)","count"), Win_Rate=("Win (Net)", lambda x: x.mean()*100),
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
                   headers=["Ticker","Title","Trades","Win(Net)%","Mean Net","Total Net"],
                   tablefmt="rounded_outline"))
                   
    print(f"\n{Fore.YELLOW}⚠  1-Day Hold Logic: Trades enter on signal day and strictly exit the following day.")
    print(f"   Spread cost = half-spread paid on entry AND half-spread paid on exit.")
    print(f"   Kalshi fee  = 0.07 × |w| × P for NO, paid once.")
    print(f"   Not financial advice.{Style.RESET_ALL}\n")
    
    if plot:
        _plot_backtest(df)

def main():
    p = argparse.ArgumentParser(description="Foundation Strategy")
    p.add_argument("--gamma",      type=float, default=GAMMA)
    p.add_argument("--lookback",   type=int,   default=LOOKBACK)
    p.add_argument("--min-volume", type=float, default=50)
    p.add_argument("--max-weight", type=float, default=MAX_WEIGHT)
    p.add_argument("--min-r2",     type=float, default=MIN_R2)
    p.add_argument("--fetch-days", type=int,   default=365)
    p.add_argument("--limit",      type=int,   default=200)
    p.add_argument("--top-n",      type=int,   default=20)
    p.add_argument("--min-price",  type=float, default=0.15)
    p.add_argument("--export",     metavar="FILE.csv")
    p.add_argument("--plot",       action="store_true")
    p.add_argument("--backtest",   action="store_true")
    args = p.parse_args()
    
    if args.backtest:
        df = run_backtest(
            gamma=args.gamma, lookback=args.lookback, min_volume=args.min_volume,
            max_weight=args.max_weight, min_r2=args.min_r2, min_price=args.min_price,
            fetch_days=args.fetch_days, limit=args.limit,
        )
        print_backtest_report(df, top_n=args.top_n, plot=args.plot)
    else:
        df = run(
            gamma=args.gamma, lookback=args.lookback, min_volume=args.min_volume,
            max_weight=args.max_weight, min_r2=args.min_r2, min_price=args.min_price,
            limit=args.limit,
        )
        print_report(df, top_n=args.top_n)

if __name__ == "__main__":
    main()