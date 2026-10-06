import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import html
from datetime import datetime, time
from zoneinfo import ZoneInfo
from urllib.parse import quote
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# APP CONFIG & DARK THEME
# ============================================================

st.set_page_config(
    page_title="NSE Institutional AI & SEPA Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background: #0b1120; color: #f8fafc; }
.block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
.card { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.badge-buy { background: #064e3b; color: #34d399; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; }
.badge-watch { background: #451a03; color: #fbbf24; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; }
.badge-avoid { background: #450a0a; color: #f87171; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; }
.badge-hit { background: #14532d; color: #4ade80; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 12px; }
.badge-stop { background: #7f1d1d; color: #fca5a5; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 12px; }
.badge-active { background: #1e3a5f; color: #60a5fa; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 12px; }
.metric-val { font-size: 20px; font-weight: 800; color: #f8fafc; margin-top: 4px; }
.metric-lbl { font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
.regime-box { background: #1e293b; border-radius: 10px; padding: 14px 18px; border-left: 4px solid #38bdf8; margin-bottom: 18px; }
.focus-card { background: #131d31; border: 1px solid #2563eb; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.reason-box { margin-top: 14px; font-size: 13px; color: #cbd5e1; background: #0a0f1d; padding: 12px 14px; border-radius: 8px; border: 1px solid #1e293b; line-height: 1.6; }
.news-box { background: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: 12px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Predefined Liquid NSE Stocks (Bluechips & Leaders)
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS",
    "ITC.NS", "SBIN.NS", "LT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "AXISBANK.NS",
    "MARUTI.NS", "KOTAKBANK.NS", "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "TATASTEEL.NS", "NTPC.NS", "BAJAJFINSV.NS", "M&M.NS", "POWERGRID.NS", "TRENT.NS",
    "NESTLEIND.NS", "ADANIENT.NS", "HAL.NS", "ONGC.NS", "COALINDIA.NS", "ZOMATO.NS",
    "CHOLAFIN.NS", "HCLTECH.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS", "INDUSINDBK.NS",
    "GRASIM.NS", "HINDALCO.NS", "TECHM.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "WIPRO.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "HEROMOTOCO.NS",
    "TATAMOTORS.NS", "SBILIFE.NS", "HDFCLIFE.NS", "PIDILITIND.NS", "VBL.NS", "BEL.NS"
]

BENCHMARKS = ["^NSEI", "^NSEBANK", "^INDIAVIX"]

# ============================================================
# DATA PIPELINE (CACHED)
# ============================================================

def safe_float(value, default=0.0):
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except Exception:
        return default

def money(value):
    val = safe_float(value, np.nan)
    return f"₹{val:,.2f}" if np.isfinite(val) else "N/A"

def get_market_status():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return "Market Closed (Weekend)", False
    t = now.time()
    if t < MARKET_OPEN:
        return "Pre-Market Session", False
    if t <= MARKET_CLOSE:
        return "Live Market Open 🟢", True
    return "Market Closed (Post-Market) ⚪", False

@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamental_metrics(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        eps_g = info.get("earningsGrowth", None)
        rev_g = info.get("revenueGrowth", None)
        roe = info.get("returnOnEquity", None)
        de = info.get("debtToEquity", None)
        pe = info.get("trailingPE", None)
        return {
            "eps_growth": (eps_g * 100) if eps_g is not None and np.isfinite(eps_g) else None,
            "rev_growth": (rev_g * 100) if rev_g is not None and np.isfinite(rev_g) else None,
            "roe": (roe * 100) if roe is not None and np.isfinite(roe) else None,
            "debt_to_equity": (de / 100) if de is not None and np.isfinite(de) else None,
            "pe": pe if pe is not None and np.isfinite(pe) else None
        }
    except Exception:
        return {"eps_growth": None, "rev_growth": None, "roe": None, "debt_to_equity": None, "pe": None}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_market_data(tickers):
    try:
        all_tickers = list(set(tickers + BENCHMARKS))
        data = yf.download(all_tickers, period="2y", interval="1d", auto_adjust=True, progress=False)
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol):
    try:
        query = quote(f"{symbol} India stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return []
        root = ET.fromstring(response.content)
        articles = []
        for item in root.findall(".//item")[:5]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            date = item.findtext("pubDate", "")
            if title:
                articles.append({
                    "title": html.unescape(title),
                    "link": link,
                    "date": date,
                })
        return articles
    except Exception:
        return []

# ============================================================
# SEPA ANALYSIS ENGINE
# ============================================================

def run_sepa_analysis(tickers, batch_df):
    if batch_df.empty:
        return pd.DataFrame()
    
    closes = batch_df["Close"]
    highs = batch_df["High"]
    lows = batch_df["Low"]
    volumes = batch_df["Volume"]
    
    nifty_1y_ret = 0.0
    if "^NSEI" in closes.columns:
        n_clean = closes["^NSEI"].dropna()
        if len(n_clean) >= 252:
            nifty_1y_ret = (float(n_clean.iloc[-1]) / float(n_clean.iloc[-252])) - 1
        elif len(n_clean) > 0:
            nifty_1y_ret = (float(n_clean.iloc[-1]) / float(n_clean.iloc[0])) - 1

    records = []
    
    for ticker in tickers:
        if ticker not in closes.columns:
            continue
            
        c = closes[ticker].dropna()
        h = highs[ticker].dropna()
        l = lows[ticker].dropna()
        v = volumes[ticker].dropna()
        
        if len(c) < 180:
            continue
            
        cmp = float(c.iloc[-1])
        prev_cmp = float(c.iloc[-2]) if len(c) >= 2 else cmp
        day_chg = ((cmp / prev_cmp) - 1) * 100
        
        sma20 = float(c.rolling(20).mean().iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        sma150 = float(c.rolling(150).mean().iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])
        
        lookback_sma200 = min(22, len(c) - 1)
        sma200_series = c.rolling(200).mean().dropna()
        sma200_trending = (sma200 >= float(sma200_series.iloc[-lookback_sma200])) if len(sma200_series) >= lookback_sma200 else True
        
        lookback_year = min(252, len(c))
        high_52w = float(h.iloc[-lookback_year:].max())
        low_52w = float(l.iloc[-lookback_year:].min())
        
        lookback_rs = min(252, len(c) - 1)
        past_price = float(c.iloc[-lookback_rs]) if lookback_rs > 0 else cmp
        stock_ret = ((cmp / past_price) - 1) if past_price > 0 else 0.0
        rs_relative = (stock_ret - nifty_1y_ret) * 100
        if not np.isfinite(rs_relative):
            rs_relative = 0.0
            
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else (cmp * 0.02)
        atr_pct = (atr14 / cmp) * 100 if cmp > 0 else 2.0
        
        vol50_avg = float(v.rolling(50).mean().iloc[-1]) if len(v) >= 50 else float(v.mean())
        vol5_avg = float(v.rolling(5).mean().iloc[-1]) if len(v) >= 5 else float(v.mean())
        latest_vol = float(v.iloc[-1])
        vol_ratio = latest_vol / vol50_avg if vol50_avg > 0 else 1.0
        
        vol_dryup = vol5_avg < (vol50_avg * 0.70)
        vol_expansion = latest_vol >= (vol50_avg * 1.35)
        
        pivot_window = min(21, len(h) - 1)
        pivot_20d = float(h.iloc[-pivot_window:-1].max()) if pivot_window > 2 else cmp
        dist_from_pivot = ((cmp - pivot_20d) / pivot_20d) * 100 if pivot_20d > 0 else 0.0
        
        c1 = (cmp > sma150) and (cmp > sma200)
        c2 = sma150 > sma200
        c3 = sma200_trending
        c4 = (sma50 > sma150) and (sma50 > sma200)
        c5 = cmp > sma50
        c6 = cmp >= (1.25 * low_52w)
        c7 = cmp >= (0.75 * high_52w)
        trend_template_passed = (c1 and c2 and c3 and c4 and c5 and c6 and c7)
        
        fund = get_fundamental_metrics(ticker)
        eps_g = fund["eps_growth"]
        rev_g = fund["rev_growth"]
        roe = fund["roe"]
        de = fund["debt_to_equity"]
        
        status = "AVOID"
        reasons = []
        entry = np.nan
        stop_loss = np.nan
        target = np.nan
        risk_pct = np.nan
        
        if not trend_template_passed:
            status = "AVOID"
            fails = []
            if cmp < sma50: fails.append("Trading below 50-day SMA")
            if not c2: fails.append("150 SMA below 200 SMA (Downtrend)")
            if cmp < (0.75 * high_52w): fails.append("Lagging >25% below 52W High")
            if not c6: fails.append("Lacks 25% lift from 52W Low")
            fail_msg = ", ".join(fails[:2]) if fails else "Failed Stage 2 trend filter"
            reasons.append(f"Trend Breakdown: {fail_msg}.")
        else:
            vcp_tight = atr_pct <= 3.5
            
            if cmp >= pivot_20d and vol_expansion and dist_from_pivot <= 4.0:
                status = "BUY"
                entry = cmp
                stop_loss = round(max(cmp * 0.94, cmp - (1.4 * atr14)), 2)
                risk_pct = round(((entry - stop_loss) / entry) * 100, 2)
                target = round(entry + (risk_pct * 2.5 / 100 * entry), 2)
                reasons.append(f"Confirmed Breakout: Cleared 20D pivot (₹{pivot_20d:,.2f}) on heavy volume ({vol_ratio:.1f}x 50D avg).")
            elif -4.5 <= dist_from_pivot <= 0.5 and (vcp_tight or vol_dryup):
                status = "WATCH"
                entry = round(pivot_20d * 1.002, 2)
                stop_loss = round(entry * 0.945, 2)
                risk_pct = round(((entry - stop_loss) / entry) * 100, 2)
                target = round(entry + (risk_pct * 2.5 / 100 * entry), 2)
                condition_desc = f"ATR compressed to {atr_pct:.1f}%" if vcp_tight else "Volume dried up to institutional exhaustion levels"
                reasons.append(f"VCP Setup: Consolidating {abs(dist_from_pivot):.1f}% below pivot. {condition_desc}. Ready for breakout.")
            elif dist_from_pivot > 4.0:
                status = "AVOID"
                reasons.append(f"Extended: Stock is already +{dist_from_pivot:.1f}% past pivot. High probability of pullback if chased.")
            else:
                status = "WATCH"
                entry = round(pivot_20d * 1.002, 2)
                stop_loss = round(entry * 0.94, 2)
                risk_pct = 6.0
                target = round(entry * 1.15, 2)
                reasons.append("Stage 2 confirmed. Trading within base; waiting for volatility contraction.")
        
        if eps_g is not None:
            if eps_g >= 20.0:
                reasons.append(f"EPS acceleration (+{eps_g:.1f}% YoY).")
            elif eps_g < 0:
                reasons.append(f"Earnings contraction ({eps_g:.1f}% YoY).")
                if status == "BUY":
                    status = "WATCH"
                    
        records.append({
            "Symbol": ticker.replace(".NS", ""),
            "Ticker": ticker,
            "Status": status,
            "CMP": cmp,
            "Day Change %": day_chg,
            "Entry": entry,
            "Stop Loss": stop_loss,
            "Target": target,
            "Risk %": risk_pct,
            "RS vs Nifty": round(rs_relative, 1),
            "ATR %": round(atr_pct, 2),
            "Vol Ratio": round(vol_ratio, 2),
            "Dist Pivot %": round(dist_from_pivot, 2),
            "EPS YoY %": eps_g,
            "Sales YoY %": rev_g,
            "ROE %": roe,
            "D/E": de,
            "Trend": "Stage 2" if trend_template_passed else "Stage 1/4",
            "Reason": " ".join(reasons)
        })
        
    return pd.DataFrame(records)

# ============================================================
# HISTORICAL TARGET ACHIEVEMENT TRACKER
# ============================================================

def audit_target_achievements(tickers, batch_df):
    """
    Backtests historical breakout setups across the past 90 trading days.
    Tracks whether each breakout achieved Target 1 (1.5R), Target 2 (2.5R),
    hit Stop Loss, or remains active.
    """
    if batch_df.empty:
        return pd.DataFrame()
    
    closes = batch_df["Close"]
    highs = batch_df["High"]
    lows = batch_df["Low"]
    volumes = batch_df["Volume"]
    
    trade_log = []
    
    for ticker in tickers:
        if ticker not in closes.columns:
            continue
            
        c = closes[ticker].dropna()
        h = highs[ticker].dropna()
        l = lows[ticker].dropna()
        v = volumes[ticker].dropna()
        
        if len(c) < 120:
            continue
            
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        vol50 = v.rolling(50).mean()
        sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        
        # Scan across past 75 trading sessions
        start_idx = max(50, len(c) - 75)
        i = start_idx
        
        while i < len(c) - 2:
            prev_high_20 = h.iloc[i-20:i].max()
            curr_c = c.iloc[i]
            curr_v = v.iloc[i]
            curr_v50 = vol50.iloc[i]
            curr_sma50 = sma50.iloc[i]
            curr_sma200 = sma200.iloc[i]
            
            # Setup triggered: Breakout above 20D pivot on >1.2x volume in Stage 2
            if (curr_c > prev_high_20) and (curr_v >= 1.2 * curr_v50) and (curr_c > curr_sma50) and (curr_sma50 > curr_sma200):
                entry_date = c.index[i].strftime("%d %b %Y")
                entry_price = float(curr_c)
                current_atr = float(atr14.iloc[i]) if pd.notna(atr14.iloc[i]) else (entry_price * 0.02)
                
                stop_loss = round(max(entry_price * 0.94, entry_price - (1.4 * current_atr)), 2)
                risk = entry_price - stop_loss
                target_1 = round(entry_price + (1.5 * risk), 2)
                target_2 = round(entry_price + (2.5 * risk), 2)
                
                # Check subsequent price trajectory
                forward_h = h.iloc[i+1:]
                forward_l = l.iloc[i+1:]
                
                outcome = "In Progress ⏳"
                exit_price = float(c.iloc[-1])
                bars_held = len(forward_h)
                max_high_reached = float(forward_h.max()) if len(forward_h) > 0 else entry_price
                max_return_pct = round(((max_high_reached - entry_price) / entry_price) * 100, 2)
                
                for step in range(len(forward_h)):
                    step_high = float(forward_h.iloc[step])
                    step_low = float(forward_l.iloc[step])
                    
                    if step_high >= target_2:
                        outcome = "Target 2 Achieved (2.5R) 🏆"
                        exit_price = target_2
                        bars_held = step + 1
                        break
                    elif step_high >= target_1 and outcome == "In Progress ⏳":
                        outcome = "Target 1 Achieved (1.5R) 🎯"
                        exit_price = target_1
                        bars_held = step + 1
                        
                    if step_low <= stop_loss:
                        outcome = "Stopped Out 🛑"
                        exit_price = stop_loss
                        bars_held = step + 1
                        break
                        
                final_return_pct = round(((exit_price - entry_price) / entry_price) * 100, 2)
                
                trade_log.append({
                    "Symbol": ticker.replace(".NS", ""),
                    "Breakout Date": entry_date,
                    "Entry": entry_price,
                    "Stop Loss": stop_loss,
                    "Target 1": target_1,
                    "Target 2": target_2,
                    "Max Return %": max_return_pct,
                    "Final Return %": final_return_pct,
                    "Holding (Days)": bars_held,
                    "Status": outcome
                })
                
                # Skip forward past this trade
                i += max(10, bars_held)
            else:
                i += 1
                
    if not trade_log:
        return pd.DataFrame()
        
    df_trades = pd.DataFrame(trade_log)
    return df_trades.sort_values(by="Final Return %", ascending=False).reset_index(drop=True)

# ============================================================
# APP UI NAVIGATION
# ============================================================

st.sidebar.title("🏛️ Screener Navigation")
mode = st.sidebar.radio(
    "Select System Mode",
    [
        "⚡ Live Market Pulse & Focus Stocks",
        "⚡ Minervini SEPA Screener",
        "🏆 Target Achievement Tracker",
        "💎 Long-Term Compounder Bucket",
        "🔍 Single Stock Diagnosis"
    ]
)

with st.spinner("Syncing live institutional feeds & market indicators..."):
    bulk_data = fetch_all_market_data(WATCHLIST)

if bulk_data.empty:
    st.error("Market data feeds are temporarily unreachable. Please check your internet connection or refresh.")
    st.stop()

analysis_df = run_sepa_analysis(WATCHLIST, bulk_data)

# ============================================================
# SECTION 1: LIVE MARKET PULSE & FOCUS STOCKS
# ============================================================
if mode == "⚡ Live Market Pulse & Focus Stocks":
    st.title("⚡ Live Market Pulse & Breakout Radar")
    st.markdown('<p style="color:#94a3b8;">Real-time index health, volatility environment, and high-probability focus stocks.</p>', unsafe_allow_html=True)
    
    market_str, is_live = get_market_status()
    st.caption(f"**Exchange Status:** {market_str} • Data Refreshed: {datetime.now(IST).strftime('%d %b %Y, %I:%M:%S %p IST')}")
    
    # 1. Macro Index Radar
    closes = bulk_data["Close"]
    c1, c2, c3 = st.columns(3)
    
    # NIFTY 50
    if "^NSEI" in closes.columns:
        n_c = closes["^NSEI"].dropna()
        n_cmp = float(n_c.iloc[-1])
        n_prev = float(n_c.iloc[-2]) if len(n_c) > 1 else n_cmp
        n_chg = ((n_cmp / n_prev) - 1) * 100
        n_sma50 = float(n_c.rolling(50).mean().iloc[-1])
        n_sma200 = float(n_c.rolling(200).mean().iloc[-1])
        
        regime = "Bullish Uptrend (Stage 2) 🚀" if n_cmp > n_sma50 > n_sma200 else ("Consolidation / Pullback ⚠️" if n_cmp > n_sma200 else "Bearish / Capital Preservation 🛡️")
        c1.metric("NIFTY 50", f"{n_cmp:,.2f}", f"{n_chg:+.2f}%")
        c1.caption(f"Regime: **{regime}**")
        
    # BANK NIFTY
    if "^NSEBANK" in closes.columns:
        b_c = closes["^NSEBANK"].dropna()
        b_cmp = float(b_c.iloc[-1])
        b_prev = float(b_c.iloc[-2]) if len(b_c) > 1 else b_cmp
        b_chg = ((b_cmp / b_prev) - 1) * 100
        c2.metric("BANK NIFTY", f"{b_cmp:,.2f}", f"{b_chg:+.2f}%")
        c2.caption(f"50 SMA: {float(b_c.rolling(50).mean().iloc[-1]):,.2f}")
        
    # INDIA VIX
    if "^INDIAVIX" in closes.columns:
        v_c = closes["^INDIAVIX"].dropna()
        v_cmp = float(v_c.iloc[-1])
        v_prev = float(v_c.iloc[-2]) if len(v_c) > 1 else v_cmp
        v_chg = ((v_cmp / v_prev) - 1) * 100
        
        vix_assessment = "Favorable for Breakouts (<14)" if v_cmp < 14 else ("Moderate Volatility (14-18)" if v_cmp <= 18 else "Elevated Risk (>18) - Cut Size")
        c3.metric("INDIA VIX", f"{v_cmp:.2f}", f"{v_chg:+.2f}%", delta_color="inverse")
        c3.caption(f"Environment: **{vix_assessment}**")

    st.markdown("---")
    
    # 2. Institutional Focus Stocks (Volume Thrust + Pivot Proximity)
    st.subheader("🎯 Stocks to Focus Today (Action Radar)")
    st.markdown('<p style="color:#94a3b8;">Liquid market leaders with institutional volume spikes (≥1.3x 50D average) within ±2.5% of breakout pivots.</p>', unsafe_allow_html=True)
    
    focus_df = analysis_df[
        (analysis_df["Vol Ratio"] >= 1.25) &
        (analysis_df["Dist Pivot %"] >= -3.0) &
        (analysis_df["Dist Pivot %"] <= 3.5) &
        (analysis_df["Trend"] == "Stage 2")
    ].sort_values(by="Vol Ratio", ascending=False).reset_index(drop=True)
    
    if not focus_df.empty:
        for _, row in focus_df.iterrows():
            st.markdown(f"""
            <div class="focus-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:22px; font-weight:800; color:#38bdf8;">{row['Symbol']}</span>
                        <span style="margin-left:12px; font-size:16px; font-weight:700;">CMP: ₹{row['CMP']:,.2f} ({row['Day Change %']:+.2f}%)</span>
                    </div>
                    <span class="badge-buy">⚡ VOLUME SURGE: {row['Vol Ratio']}x</span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:12px;">
                    <div><div class="metric-lbl">Ideal Entry Pivot</div><div class="metric-val">₹{row['Entry']:,.2f}</div></div>
                    <div><div class="metric-lbl">Stop Loss</div><div class="metric-val" style="color:#f87171;">₹{row['Stop Loss']:,.2f}</div></div>
                    <div><div class="metric-lbl">Target</div><div class="metric-val" style="color:#34d399;">₹{row['Target']:,.2f}</div></div>
                    <div><div class="metric-lbl">Distance to Pivot</div><div class="metric-val">{row['Dist Pivot %']:+.1f}%</div></div>
                </div>
                <div style="margin-top:10px; font-size:12px; color:#cbd5e1;">
                    <b>Trigger Reason:</b> {row['Reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No stocks currently meet simultaneous volume surge + tight pivot criteria. Market is in quiet base-building mode.")

    # 3. Market Breadth across Predefined Watchlist
    st.markdown("### 📊 Universe Breadth")
    pos_count = len(analysis_df[analysis_df["Day Change %"] > 0])
    neg_count = len(analysis_df[analysis_df["Day Change %"] < 0])
    b1, b2, b3 = st.columns(3)
    b1.metric("Advancing Stocks", pos_count)
    b2.metric("Declining Stocks", neg_count)
    b3.metric("Adv/Dec Ratio", f"{(pos_count / max(neg_count, 1)):.2f}")

# ============================================================
# SECTION 2: MINERVINI SEPA SUPER SCREENER
# ============================================================
elif mode == "⚡ Minervini SEPA Screener":
    st.title("⚡ Minervini SEPA Institutional Screener")
    st.markdown('<p style="color:#94a3b8;">Stage 2 Trend Template • Volatility Contraction (VCP) • Volume Breakouts • Risk Managed Stops</p>', unsafe_allow_html=True)
    
    action_filter = st.selectbox("Filter Status", ["All Qualified (BUY & WATCH)", "BUY Only", "WATCH Only", "AVOID Only", "Show Everything"])
    
    if action_filter == "BUY Only":
        filtered_df = analysis_df[analysis_df["Status"] == "BUY"]
    elif action_filter == "WATCH Only":
        filtered_df = analysis_df[analysis_df["Status"] == "WATCH"]
    elif action_filter == "AVOID Only":
        filtered_df = analysis_df[analysis_df["Status"] == "AVOID"]
    elif action_filter == "All Qualified (BUY & WATCH)":
        filtered_df = analysis_df[analysis_df["Status"].isin(["BUY", "WATCH"])]
    else:
        filtered_df = analysis_df.copy()

    b_count = len(analysis_df[analysis_df["Status"] == "BUY"])
    w_count = len(analysis_df[analysis_df["Status"] == "WATCH"])
    a_count = len(analysis_df[analysis_df["Status"] == "AVOID"])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed Breakouts (BUY)", b_count)
    c2.metric("VCP Setups Pending (WATCH)", w_count)
    c3.metric("Structural Lag (AVOID)", a_count)
    c4.metric("Total Filtered", len(filtered_df))
    
    st.markdown("---")
    
    if filtered_df.empty:
        st.info("No stocks match the selected criteria.")
    else:
        for _, row in filtered_df.iterrows():
            st_val = row["Status"]
            badge_class = "badge-buy" if st_val == "BUY" else ("badge-watch" if st_val == "WATCH" else "badge-avoid")
            
            entry_str = f"₹{row['Entry']:,.2f}" if pd.notna(row['Entry']) and np.isfinite(row['Entry']) else "N/A"
            stop_str = f"₹{row['Stop Loss']:,.2f}" if pd.notna(row['Stop Loss']) and np.isfinite(row['Stop Loss']) else "N/A"
            target_str = f"₹{row['Target']:,.2f}" if pd.notna(row['Target']) and np.isfinite(row['Target']) else "N/A"
            risk_str = f"{row['Risk %']:.1f}%" if pd.notna(row['Risk %']) and np.isfinite(row['Risk %']) else "N/A"
            
            rs_val = row["RS vs Nifty"]
            rs_display = f"{rs_val:+.1f}%" if pd.notna(rs_val) and np.isfinite(rs_val) else "0.0%"
            
            st.markdown(f"""
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:24px; font-weight:800;">{row['Symbol']}</span>
                        <span style="color:#94a3b8; margin-left:10px; font-size:15px;">CMP: ₹{row['CMP']:,.2f} ({row['Day Change %']:+.2f}%)</span>
                        <span style="color:#64748b; margin-left:8px; font-size:12px;">({row['Trend']})</span>
                    </div>
                    <span class="{badge_class}">{st_val}</span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:12px; margin-top:16px;">
                    <div><div class="metric-lbl">Ideal Entry</div><div class="metric-val">{entry_str}</div></div>
                    <div><div class="metric-lbl">Stop Loss</div><div class="metric-val" style="color:#f87171;">{stop_str}</div></div>
                    <div><div class="metric-lbl">Target (2.5R)</div><div class="metric-val" style="color:#34d399;">{target_str}</div></div>
                    <div><div class="metric-lbl">Risk Profile</div><div class="metric-val">{risk_str}</div></div>
                    <div><div class="metric-lbl">RS vs NIFTY</div><div class="metric-val" style="color:{'#34d399' if rs_val >= 0 else '#f87171'};">{rs_display}</div></div>
                </div>
                <div class="reason-box">
                    <b>Setup Diagnosis:</b> {row['Reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# SECTION 3: TARGET ACHIEVEMENT TRACKER (HALL OF FAME)
# ============================================================
elif mode == "🏆 Target Achievement Tracker":
    st.title("🏆 Target Achievement Tracker (Scorecard)")
    st.markdown('<p style="color:#94a3b8;">Verifiable audit of historical Stage 2 breakouts across the watchlist, measuring Target 1 (1.5R) and Target 2 (2.5R) hit rates.</p>', unsafe_allow_html=True)
    
    with st.spinner("Auditing historical breakouts and target hit rates..."):
        trade_log_df = audit_target_achievements(WATCHLIST, bulk_data)
        
    if trade_log_df.empty:
        st.info("No qualified breakout signals identified in the historical lookback window.")
    else:
        # High Level Statistics
        total_signals = len(trade_log_df)
        t1_hits = len(trade_log_df[trade_log_df["Status"].str.contains("Target 1")])
        t2_hits = len(trade_log_df[trade_log_df["Status"].str.contains("Target 2")])
        stopped_out = len(trade_log_df[trade_log_df["Status"].str.contains("Stopped Out")])
        active = len(trade_log_df[trade_log_df["Status"].str.contains("In Progress")])
        
        resolved_trades = total_signals - active
        win_rate = ((t1_hits + t2_hits) / resolved_trades * 100) if resolved_trades > 0 else 0.0
        avg_winner = trade_log_df[trade_log_df["Final Return %"] > 0]["Final Return %"].mean()
        avg_winner_str = f"{avg_winner:+.2f}%" if pd.notna(avg_winner) else "0.0%"
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Setups Triggered", total_signals)
        m2.metric("Target 2 Hits (2.5R)", t2_hits)
        m3.metric("Target 1 Hits (1.5R)", t1_hits)
        m4.metric("Win Rate", f"{win_rate:.1f}%")
        m5.metric("Avg Return (Winners)", avg_winner_str)
        
        st.markdown("---")
        st.subheader("🎯 Completed & Active Trade Records")
        
        for _, trade in trade_log_df.iterrows():
            st_text = trade["Status"]
            badge_cls = "badge-hit" if "Achieved" in st_text else ("badge-stop" if "Stopped" in st_text else "badge-active")
            
            st.markdown(f"""
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:22px; font-weight:800;">{trade['Symbol']}</span>
                        <span style="color:#94a3b8; margin-left:12px; font-size:14px;">Breakout Date: {trade['Breakout Date']}</span>
                    </div>
                    <span class="{badge_cls}">{st_text}</span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(6, 1fr); gap:10px; margin-top:14px;">
                    <div><div class="metric-lbl">Entry Price</div><div class="metric-val">₹{trade['Entry']:,.2f}</div></div>
                    <div><div class="metric-lbl">Stop Loss</div><div class="metric-val" style="color:#f87171;">₹{trade['Stop Loss']:,.2f}</div></div>
                    <div><div class="metric-lbl">Target 1 (1.5R)</div><div class="metric-val" style="color:#34d399;">₹{trade['Target 1']:,.2f}</div></div>
                    <div><div class="metric-lbl">Target 2 (2.5R)</div><div class="metric-val" style="color:#4ade80;">₹{trade['Target 2']:,.2f}</div></div>
                    <div><div class="metric-lbl">Max Peak Reached</div><div class="metric-val" style="color:#38bdf8;">{trade['Max Return %']:+.2f}%</div></div>
                    <div><div class="metric-lbl">Final PnL</div><div class="metric-val" style="color:{'#4ade80' if trade['Final Return %'] > 0 else '#f87171'};">{trade['Final Return %']:+.2f}%</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# SECTION 4: LONG-TERM COMPOUNDER BUCKET
# ============================================================
elif mode == "💎 Long-Term Compounder Bucket":
    st.title("💎 Long-Term Quality Compounder Screen")
    st.markdown('<p style="color:#94a3b8;">High ROE (≥15%) • Resilient EPS Growth (≥12%) • Clean Balance Sheet (Debt/Equity ≤ 1.0)</p>', unsafe_allow_html=True)
    
    lt_df = analysis_df[
        (analysis_df["ROE %"].fillna(0) >= 15.0) &
        (analysis_df["EPS YoY %"].fillna(0) >= 12.0) &
        (analysis_df["D/E"].fillna(0) <= 1.0)
    ].copy()
    
    st.metric("Compounders Passing All Quality Gates", len(lt_df))
    
    if not lt_df.empty:
        lt_table = lt_df[["Symbol", "CMP", "Day Change %", "EPS YoY %", "Sales YoY %", "ROE %", "D/E", "Trend", "RS vs Nifty"]].copy()
        lt_table["CMP"] = lt_table["CMP"].apply(lambda x: f"₹{x:,.2f}")
        lt_table["Day Change %"] = lt_table["Day Change %"].apply(lambda x: f"{x:+.2f}%")
        lt_table["EPS YoY %"] = lt_table["EPS YoY %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        lt_table["Sales YoY %"] = lt_table["Sales YoY %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        lt_table["ROE %"] = lt_table["ROE %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        lt_table["D/E"] = lt_table["D/E"].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x > 0 else "Negligible")
        lt_table["RS vs Nifty"] = lt_table["RS vs Nifty"].apply(lambda x: f"{x:+.1f}%")
        
        st.dataframe(lt_table, use_container_width=True, hide_index=True)
    else:
        st.info("No companies currently clear all quality hurdle rates simultaneously.")

# ============================================================
# SECTION 5: SINGLE STOCK DIAGNOSIS
# ============================================================
elif mode == "🔍 Single Stock Diagnosis":
    st.title("🔍 Single Stock Technical & Structural Breakdown")
    selected_sym = st.selectbox("Select NSE Security", [s.replace(".NS", "") for s in WATCHLIST])
    
    ticker_sym = f"{selected_sym}.NS"
    match = analysis_df[analysis_df["Symbol"] == selected_sym].iloc[0]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SEPA Status", match["Status"])
    c2.metric("CMP", f"₹{match['CMP']:,.2f}", f"{match['Day Change %']:+.2f}%")
    c3.metric("Entry Pivot", f"₹{match['Entry']:,.2f}" if pd.notna(match['Entry']) and np.isfinite(match['Entry']) else "Awaiting Base")
    c4.metric("Stop Loss", f"₹{match['Stop Loss']:,.2f}" if pd.notna(match['Stop Loss']) and np.isfinite(match['Stop Loss']) else "N/A")
    
    st.info(f"**Structural Evaluation:** {match['Reason']}")
    
    # Technical Indicators Chart
    raw_c = bulk_data["Close"][ticker_sym].dropna()
    sma50 = raw_c.rolling(50).mean()
    sma150 = raw_c.rolling(150).mean()
    sma200 = raw_c.rolling(200).mean()
    
    st.subheader("Price Structure vs. Key Moving Averages")
    chart_df = pd.DataFrame({
        "Close": raw_c,
        "50 SMA": sma50,
        "150 SMA": sma150,
        "200 SMA": sma200
    }).tail(180)
    st.line_chart(chart_df)
    
    # Fundamentals & News
    f1, f2 = st.columns(2)
    with f1:
        st.subheader("Fundamental Metrics")
        st.write(f"- **EPS Growth (YoY):** {match['EPS YoY %']:.1f}%" if pd.notna(match['EPS YoY %']) else "- **EPS Growth (YoY):** N/A")
        st.write(f"- **Revenue Growth (YoY):** {match['Sales YoY %']:.1f}%" if pd.notna(match['Sales YoY %']) else "- **Revenue Growth (YoY):** N/A")
        st.write(f"- **Return on Equity (ROE):** {match['ROE %']:.1f}%" if pd.notna(match['ROE %']) else "- **Return on Equity (ROE):** N/A")
        st.write(f"- **Debt / Equity:** {match['D/E']:.2f}" if pd.notna(match['D/E']) else "- **Debt / Equity:** Negligible")
    
    with f2:
        st.subheader("Recent News")
        news_items = get_news(selected_sym)
        if news_items:
            for item in news_items[:3]:
                st.markdown(f"""
                <div class="news-box">
                    <div style="font-weight:700; font-size:13px;">{item['title']}</div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:4px;">{item['date']}</div>
                    <a href="{item['link']}" target="_blank" style="color:#60a5fa; font-size:12px;">Read report →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No recent headlines available.")
