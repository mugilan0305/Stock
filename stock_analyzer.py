import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
.card { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; margin-bottom: 14px; }
.badge-buy { background: #064e3b; color: #34d399; font-weight: 700; padding: 4px 10px; border-radius: 6px; }
.badge-watch { background: #451a03; color: #fbbf24; font-weight: 700; padding: 4px 10px; border-radius: 6px; }
.badge-avoid { background: #450a0a; color: #f87171; font-weight: 700; padding: 4px 10px; border-radius: 6px; }
.metric-val { font-size: 26px; font-weight: 800; color: #f8fafc; }
.metric-lbl { font-size: 12px; color: #94a3b8; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

IST = ZoneInfo("Asia/Kolkata")

# Predefined Liquid NSE Stocks (Bluechips & Market Leaders)
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

# ============================================================
# DATA PIPELINE (CACHED)
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamental_metrics(ticker):
    """Pulls fundamental data from balance sheet and earnings."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        
        eps_growth = info.get("earningsGrowth", None)
        rev_growth = info.get("revenueGrowth", None)
        roe = info.get("returnOnEquity", None)
        debt_to_equity = info.get("debtToEquity", None)
        pe_ratio = info.get("trailingPE", None)
        
        return {
            "eps_growth": (eps_growth * 100) if eps_growth else None,
            "rev_growth": (rev_growth * 100) if rev_growth else None,
            "roe": (roe * 100) if roe else None,
            "debt_to_equity": (debt_to_equity / 100) if debt_to_equity else None,
            "pe": pe_ratio if pe_ratio else None
        }
    except Exception:
        return {"eps_growth": None, "rev_growth": None, "roe": None, "debt_to_equity": None, "pe": None}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_market_data(tickers):
    """Downloads all price data simultaneously in one vectorized batch."""
    try:
        data = yf.download(tickers + ["^NSEI"], period="2y", interval="1d", auto_adjust=True, progress=False)
        return data
    except Exception:
        return pd.DataFrame()

# ============================================================
# MINERVINI SEPA & LONG-TERM ENGINE
# ============================================================

def run_sepa_analysis(tickers, batch_df):
    if batch_df.empty:
        return pd.DataFrame()
    
    closes = batch_df["Close"]
    highs = batch_df["High"]
    lows = batch_df["Low"]
    volumes = batch_df["Volume"]
    
    # Benchmark returns for RS calculation
    nifty_close = closes["^NSEI"]
    nifty_1y_ret = (nifty_close.iloc[-1] / nifty_close.iloc[-252]) - 1
    
    records = []
    
    for ticker in tickers:
        if ticker not in closes.columns:
            continue
            
        c = closes[ticker].dropna()
        h = highs[ticker].dropna()
        l = lows[ticker].dropna()
        v = volumes[ticker].dropna()
        
        if len(c) < 252:
            continue
            
        cmp = float(c.iloc[-1])
        
        # 1. Moving Averages
        sma50 = c.rolling(50).mean().iloc[-1]
        sma150 = c.rolling(150).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        sma200_22d_ago = c.rolling(200).mean().iloc[-22]
        
        # 2. 52-Week High & Low
        high_52w = float(h.iloc[-252:].max())
        low_52w = float(l.iloc[-252:].min())
        
        # 3. Relative Strength (Stock 1Y vs Nifty 1Y)
        stock_1y_ret = (cmp / float(c.iloc[-252])) - 1
        rs_relative = (stock_1y_ret - nifty_1y_ret) * 100
        
        # 4. Volatility & Contraction (VCP metrics)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = (atr14 / cmp) * 100
        
        # Volume metrics
        vol50_avg = float(v.rolling(50).mean().iloc[-1])
        vol5_avg = float(v.rolling(5).mean().iloc[-1])
        latest_vol = float(v.iloc[-1])
        vol_dryup = vol5_avg < (vol50_avg * 0.70)
        vol_expansion = latest_vol >= (vol50_avg * 1.40)
        
        # Pivot point (Highest high of the past 20 sessions)
        pivot_20d = float(h.iloc[-21:-1].max())
        dist_from_pivot = ((cmp - pivot_20d) / pivot_20d) * 100
        
        # 5. Trend Template Checklist (7 Conditions)
        c1 = cmp > sma150 and cmp > sma200
        c2 = sma150 > sma200
        c3 = sma200 > sma200_22d_ago
        c4 = sma50 > sma150 and sma50 > sma200
        c5 = cmp > sma50
        c6 = cmp >= (1.30 * low_52w)
        c7 = cmp >= (0.75 * high_52w)
        trend_template_passed = (c1 and c2 and c3 and c4 and c5 and c6 and c7)
        
        # Fundamentals
        fund = get_fundamental_metrics(ticker)
        eps_g = fund["eps_growth"]
        rev_g = fund["rev_growth"]
        roe = fund["roe"]
        de = fund["debt_to_equity"]
        
        # Fundamental Pass checks
        fund_pass_lt = (
            (eps_g is not None and eps_g >= 15.0) and
            (rev_g is not None and rev_g >= 12.0) and
            (roe is not None and roe >= 15.0) and
            (de is None or de <= 1.2)
        )
        
        # Classification Engine
        status = "AVOID"
        reason = []
        entry = pivot_20d
        stop_loss = 0.0
        target = 0.0
        risk_pct = 0.0
        
        # Check Trend Failure
        if not trend_template_passed:
            status = "AVOID"
            failed_reasons = []
            if cmp < sma50: failed_reasons.append("Trading below 50-day SMA")
            if not c2: failed_reasons.append("150 SMA below 200 SMA (Downtrend)")
            if cmp < (0.75 * high_52w): failed_reasons.append("Lacks relative high proximity")
            reason.append("Failed Trend Template: " + ", ".join(failed_reasons[:2]))
        else:
            # Stage 2 Confirmed -> Now evaluate VCP and Breakout
            vcp_tight = atr_pct <= 3.5
            
            if cmp >= pivot_20d and vol_expansion and dist_from_pivot <= 4.0:
                status = "BUY"
                entry = cmp
                stop_loss = round(max(cmp * 0.94, cmp - (1.5 * atr14)), 2)
                risk_pct = round(((entry - stop_loss) / entry) * 100, 2)
                target = round(entry + (risk_pct * 2.5 / 100 * entry), 2)
                reason.append(f"Confirmed Breakout: Cleared 20D pivot ({pivot_20d:.2f}) with volume expansion ({latest_vol/vol50_avg:.1f}x avg).")
            elif -4.0 <= dist_from_pivot <= 0.5 and (vcp_tight or vol_dryup):
                status = "WATCH"
                entry = round(pivot_20d * 1.002, 2)
                stop_loss = round(entry * 0.945, 2)
                risk_pct = round(((entry - stop_loss) / entry) * 100, 2)
                target = round(entry + (risk_pct * 2.5 / 100 * entry), 2)
                vcp_detail = "ATR compressed to {:.1f}%".format(atr_pct) if vcp_tight else "Volume drying up (selling exhaustion)"
                reason.append(f"VCP Setup: Stock within {abs(dist_from_pivot):.1f}% of pivot with {vcp_detail}. Awaiting volume thrust.")
            elif dist_from_pivot > 4.5:
                status = "AVOID"
                reason.append(f"Extended: Stock is +{dist_from_pivot:.1f}% above pivot. Chasing increases adverse risk profile.")
            else:
                status = "WATCH"
                entry = pivot_20d
                stop_loss = round(entry * 0.94, 2)
                risk_pct = 6.0
                target = round(entry * 1.15, 2)
                reason.append("Consolidating within Stage 2 base; monitor for tighter volatility contraction.")
        
        # Append fundamental context to reason
        if eps_g and eps_g > 20:
            reason.append(f"Strong EPS growth (+{eps_g:.1f}% YoY).")
        elif eps_g and eps_g < 0:
            reason.append(f"Declining EPS ({eps_g:.1f}% YoY).")
            if status == "BUY": status = "WATCH"
            
        records.append({
            "Symbol": ticker.replace(".NS", ""),
            "Status": status,
            "CMP": cmp,
            "Entry": entry if status != "AVOID" else np.nan,
            "Stop Loss": stop_loss if status != "AVOID" else np.nan,
            "Target": target if status != "AVOID" else np.nan,
            "Risk %": risk_pct if status != "AVOID" else np.nan,
            "RS vs Nifty": round(rs_relative, 1),
            "ATR %": round(atr_pct, 2),
            "EPS YoY %": eps_g,
            "Sales YoY %": rev_g,
            "ROE %": roe,
            "D/E": de,
            "Trend": "Stage 2" if trend_template_passed else "Stage 1/4",
            "Reason": " • ".join(reason)
        })
        
    return pd.DataFrame(records)

# ============================================================
# APP UI NAVIGATION
# ============================================================

st.sidebar.title("🏛️ Screener Controls")
mode = st.sidebar.radio(
    "Choose Framework",
    ["⚡ Minervini SEPA Screener", "💎 Long-Term Compounder Bucket", "🔍 Single Stock Diagnosis"]
)

with st.spinner("Downloading market matrices..."):
    bulk_data = fetch_all_market_data(WATCHLIST)

if bulk_data.empty:
    st.error("Market data feeds are temporarily unreachable. Verify network connectivity.")
    st.stop()

analysis_df = run_sepa_analysis(WATCHLIST, bulk_data)

# ============================================================
# TAB 1: MINERVINI SEPA SUPER SCREENER
# ============================================================
if mode == "⚡ Minervini SEPA Screener":
    st.title("⚡ Minervini SEPA Institutional Screener")
    st.markdown('<p style="color:#94a3b8;">Filters for Volatility Contraction Patterns (VCP), institutional volume breakouts, and Stage 2 trend structure.</p>', unsafe_allow_html=True)
    
    # Filter Tabs
    action_filter = st.selectbox("Filter Status", ["All Qualified", "BUY Now", "WATCH Setup", "AVOID"])
    
    filtered_df = analysis_df.copy()
    if action_filter == "BUY Now":
        filtered_df = filtered_df[filtered_df["Status"] == "BUY"]
    elif action_filter == "WATCH Setup":
        filtered_df = filtered_df[filtered_df["Status"] == "WATCH"]
    elif action_filter == "AVOID":
        filtered_df = filtered_df[filtered_df["Status"] == "AVOID"]
    else:
        filtered_df = filtered_df[filtered_df["Status"].isin(["BUY", "WATCH"])]

    # Metrics Summary
    b_count = len(analysis_df[analysis_df["Status"] == "BUY"])
    w_count = len(analysis_df[analysis_df["Status"] == "WATCH"])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Confirmed Breakouts (BUY)", b_count)
    c2.metric("VCP Setups Pending (WATCH)", w_count)
    c3.metric("Total Analyzed", len(analysis_df))
    
    st.markdown("### Actionable Trade Setups")
    
    for _, row in filtered_df.iterrows():
        badge_class = "badge-buy" if row["Status"] == "BUY" else ("badge-watch" if row["Status"] == "WATCH" else "badge-avoid")
        
        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:22px; font-weight:800;">{row['Symbol']}</span>
                    <span style="color:#94a3b8; margin-left:8px;">CMP: ₹{row['CMP']:,.2f}</span>
                </div>
                <span class="{badge_class}">{row['Status']}</span>
            </div>
            <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:10px; margin-top:14px;">
                <div><div class="metric-lbl">Optimal Entry</div><div class="metric-val" style="font-size:18px;">{'₹' + str(row['Entry']) if pd.notna(row['Entry']) else 'N/A'}</div></div>
                <div><div class="metric-lbl">Stop Loss</div><div class="metric-val" style="font-size:18px; color:#f87171;">{'₹' + str(row['Stop Loss']) if pd.notna(row['Stop Loss']) else 'N/A'}</div></div>
                <div><div class="metric-lbl">Target (2.5R)</div><div class="metric-val" style="font-size:18px; color:#34d399;">{'₹' + str(row['Target']) if pd.notna(row['Target']) else 'N/A'}</div></div>
                <div><div class="metric-lbl">Risk Profile</div><div class="metric-val" style="font-size:18px;">{str(row['Risk %']) + '%' if pd.notna(row['Risk %']) else 'N/A'}</div></div>
                <div><div class="metric-lbl">RS vs Nifty</div><div class="metric-val" style="font-size:18px;">{row['RS vs Nifty']:+.1f}%</div></div>
            </div>
            <div style="margin-top:12px; font-size:13px; color:#cbd5e1; background:#0f172a; padding:10px; border-radius:8px;">
                <b>Rationale:</b> {row['Reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2: LONG-TERM COMPOUNDER BUCKET
# ============================================================
elif mode == "💎 Long-Term Compounder Bucket":
    st.title("💎 Long-Term Quality Compounder Screen")
    st.markdown('<p style="color:#94a3b8;">High-quality, low-leverage companies with sustained return on equity and superior compound annual growth metrics.</p>', unsafe_allow_html=True)
    
    # Filter for long term quality
    lt_df = analysis_df[
        (analysis_df["ROE %"] >= 15.0) &
        (analysis_df["EPS YoY %"] >= 15.0) &
        ((analysis_df["D/E"] <= 1.0) | (analysis_df["D/E"].isna()))
    ].copy()
    
    st.metric("Compounders Meeting Quality Threshold", len(lt_df))
    
    if not lt_df.empty:
        table_data = lt_df[["Symbol", "CMP", "EPS YoY %", "Sales YoY %", "ROE %", "D/E", "Trend", "RS vs Nifty"]].copy()
        table_data["CMP"] = table_data["CMP"].apply(lambda x: f"₹{x:,.2f}")
        table_data["EPS YoY %"] = table_data["EPS YoY %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        table_data["Sales YoY %"] = table_data["Sales YoY %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        table_data["ROE %"] = table_data["ROE %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        table_data["D/E"] = table_data["D/E"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "Negligible")
        table_data["RS vs Nifty"] = table_data["RS vs Nifty"].apply(lambda x: f"{x:+.1f}%")
        
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.info("No stocks currently meet all combined long-term fundamental hurdles.")

# ============================================================
# TAB 3: SINGLE STOCK DIAGNOSIS
# ============================================================
elif mode == "🔍 Single Stock Diagnosis":
    st.title("🔍 Single Stock Technical & Structural Breakdown")
    selected_sym = st.selectbox("Select Security", [s.replace(".NS", "") for s in WATCHLIST])
    
    ticker_sym = f"{selected_sym}.NS"
    match = analysis_df[analysis_df["Symbol"] == selected_sym].iloc[0]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", match["Status"])
    c2.metric("CMP", f"₹{match['CMP']:,.2f}")
    c3.metric("Optimal Entry", f"₹{match['Entry']:,.2f}" if pd.notna(match['Entry']) else "Wait")
    c4.metric("Stop Loss", f"₹{match['Stop Loss']:,.2f}" if pd.notna(match['Stop Loss']) else "N/A")
    
    st.info(f"**Structural Rationale:** {match['Reason']}")
    
    # Charting price vs SMAs
    raw_c = bulk_data["Close"][ticker_sym].dropna()
    chart_df = pd.DataFrame({
        "Close": raw_c,
        "50 SMA": raw_c.rolling(50).mean(),
        "150 SMA": raw_c.rolling(150).mean(),
        "200 SMA": raw_c.rolling(200).mean()
    }).tail(180)
    
    st.line_chart(chart_df)
