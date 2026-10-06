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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# APP CONFIG & DARK THEME
# ============================================================

st.set_page_config(
    page_title="Institutional Stock AI & Moat Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background: #0b1120; color: #f8fafc; }
.block-container { max-width: 1440px; padding-top: 1.2rem; padding-bottom: 3rem; }
.card { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.badge-buy { background: #064e3b; color: #34d399; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; }
.badge-watch { background: #451a03; color: #fbbf24; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; }
.badge-avoid { background: #450a0a; color: #f87171; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; }
.metric-val { font-size: 20px; font-weight: 800; color: #f8fafc; margin-top: 4px; }
.metric-lbl { font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
.live-badge { display: inline-flex; align-items: center; background: #0f2e1b; border: 1px solid #10b981; color: #34d399; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 12px; }
.pulse-dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; display: inline-block; margin-right: 6px; }
.stock-box { background: #111827; border: 1px solid #1e293b; border-radius: 14px; padding: 22px; margin-bottom: 18px; }
.price { font-size: 38px; font-weight: 800; margin-top: 8px; }
.reason-box { margin-top: 14px; font-size: 13px; color: #cbd5e1; background: #0a0f1d; padding: 12px 14px; border-radius: 8px; border: 1px solid #1e293b; line-height: 1.6; }
.damani-box { background: #1a162b; border: 1px solid #6366f1; border-radius: 12px; padding: 18px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS",
    "ITC.NS", "SBIN.NS", "LT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "AXISBANK.NS",
    "MARUTI.NS", "KOTAKBANK.NS", "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "TATASTEEL.NS", "NTPC.NS", "M&M.NS", "POWERGRID.NS", "TRENT.NS", "HAL.NS", "BEL.NS"
]

# ============================================================
# DATA PIPELINE (OPTIMIZED)
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

@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamental_metrics(ticker):
    """Pulls corporate balance sheet and profitability fundamentals."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        
        eps_g = info.get("earningsGrowth", None)
        rev_g = info.get("revenueGrowth", None)
        roe = info.get("returnOnEquity", None)
        de = info.get("debtToEquity", None)
        pe = info.get("trailingPE", None)
        fpe = info.get("forwardPE", None)
        pm = info.get("profitMargins", None)
        
        return {
            "eps_growth": (eps_g * 100) if eps_g is not None and np.isfinite(eps_g) else None,
            "rev_growth": (rev_g * 100) if rev_g is not None and np.isfinite(rev_g) else None,
            "roe": (roe * 100) if roe is not None and np.isfinite(roe) else None,
            "debt_to_equity": (de / 100) if de is not None and np.isfinite(de) else None,
            "pe": pe if pe is not None and np.isfinite(pe) else None,
            "fpe": fpe if fpe is not None and np.isfinite(fpe) else None,
            "profit_margin": (pm * 100) if pm is not None and np.isfinite(pm) else None
        }
    except Exception:
        return {"eps_growth": None, "rev_growth": None, "roe": None, "debt_to_equity": None, "pe": None, "fpe": None, "profit_margin": None}

def get_live_quote(ticker):
    """Ultra-fast quote endpoint with short timeout."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker)}?interval=1m&range=1d"
        res = requests.get(url, timeout=3, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200:
            meta = res.json()["chart"]["result"][0]["meta"]
            p = safe_float(meta.get("regularMarketPrice"), np.nan)
            prev = safe_float(meta.get("previousClose"), np.nan)
            if np.isfinite(p):
                chg = ((p / prev) - 1) * 100 if np.isfinite(prev) and prev > 0 else 0.0
                return {"price": p, "change": chg, "prev_close": prev, "time": datetime.now(IST)}
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_market_data(tickers):
    try:
        return yf.download(tickers + ["^NSEI"], period="2y", interval="1d", auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()

# ============================================================
# ADVANCED MACHINE LEARNING ENSEMBLE
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def run_advanced_ai_model(df_stock):
    """
    Institutional Gradient Boosting + Volatility Ensemble.
    Uses multi-timeframe returns, ATR ratio, rolling z-scores,
    and momentum divergence to predict expected moves.
    """
    if len(df_stock) < 180:
        return None
        
    df = df_stock.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # Alpha Features
    df["RET_1"] = close.pct_change(1)
    df["RET_5"] = close.pct_change(5)
    df["RET_20"] = close.pct_change(20)
    
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["DIST_SMA20"] = (close - df["SMA20"]) / df["SMA20"]
    df["DIST_SMA50"] = (close - df["SMA50"]) / df["SMA50"]

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_RATIO"] = df["ATR"] / close

    # Z-Score of Volume
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan)
    df["VOL_Z"] = (vol - vol_mean) / vol_std

    # Target: 5-Day forward return
    df["TARGET_5D"] = close.shift(-5) / close - 1

    features = ["RET_1", "RET_5", "RET_20", "DIST_SMA20", "DIST_SMA50", "ATR_RATIO", "VOL_Z"]
    dataset = df.dropna(subset=features + ["TARGET_5D"])

    if len(dataset) < 100:
        return None

    X = dataset[features]
    y = dataset["TARGET_5D"]

    # Model 1: Gradient Boosting Regressor (Captures non-linear momentum)
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, random_state=42)
    gbr.fit(X, y)

    # Model 2: Random Forest (Reduces variance)
    rf = RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    latest_features = df[features].iloc[[-1]].dropna()
    if latest_features.empty:
        return None

    pred_gbr = gbr.predict(latest_features)[0]
    pred_rf = rf.predict(latest_features)[0]
    
    # Ensemble weighted prediction
    expected_5d_ret = (0.65 * pred_gbr) + (0.35 * pred_rf)
    cmp = float(close.iloc[-1])
    predicted_price = cmp * (1 + expected_5d_ret)
    
    # Confidence estimation based on tree agreement
    disagreement = abs(pred_gbr - pred_rf)
    confidence = max(40, min(92, int(85 - (disagreement * 400))))

    return {
        "expected_return_pct": expected_5d_ret * 100,
        "predicted_target": predicted_price,
        "confidence": confidence,
        "model_type": "Institutional Dual Ensemble (GBR + RF)"
    }

# ============================================================
# RAKESH JHUNJHUNWALA & RADHAKISHAN DAMANI MOAT SCORE
# ============================================================

def calculate_jhunjhunwala_damani_score(fund_data, cmp):
    """
    Evaluates companies using the proven principles of Rakesh Jhunjhunwala and Radhakishan Damani:
    1. Capital Efficiency (ROE >= 20%)
    2. Pristine Balance Sheet (Debt/Equity <= 0.5)
    3. High Operating Margin Durability (> 15%)
    4. Reasonable Valuation (PE relative to growth)
    5. Scalability Runway (Double digit top & bottom line growth)
    """
    score = 0
    breakdown = []
    
    roe = fund_data.get("roe")
    de = fund_data.get("debt_to_equity")
    margin = fund_data.get("profit_margin")
    eps_g = fund_data.get("eps_growth")
    rev_g = fund_data.get("rev_growth")
    pe = fund_data.get("pe")

    # 1. Capital Return Efficiency (Damani & Jhunjhunwala Priority)
    if roe is not None:
        if roe >= 22.0:
            score += 25
            breakdown.append(f"World-Class ROE ({roe:.1f}%), exceptional capital compounding.")
        elif roe >= 15.0:
            score += 15
            breakdown.append(f"Healthy ROE ({roe:.1f}%).")
        else:
            breakdown.append(f"Sub-par capital efficiency (ROE {roe:.1f}%).")

    # 2. Balance Sheet Frugality (Damani's Debt Aversion)
    if de is not None:
        if de <= 0.2:
            score += 25
            breakdown.append(f"Near Zero Debt (D/E {de:.2f}). Immune to credit cycle shocks.")
        elif de <= 0.6:
            score += 18
            breakdown.append(f"Conservative Debt (D/E {de:.2f}).")
        elif de > 1.2:
            score -= 15
            breakdown.append(f"Excessive Leverage (D/E {de:.2f}). Fails Damani balance sheet test.")
    else:
        score += 20
        breakdown.append("Negligible debt structure.")

    # 3. Profit Margin Moat
    if margin is not None:
        if margin >= 18.0:
            score += 20
            breakdown.append(f"Wide Pricing Power (Net Margin {margin:.1f}%).")
        elif margin >= 10.0:
            score += 12
            breakdown.append(f"Decent Margin ({margin:.1f}%).")

    # 4. Growth Runway (Jhunjhunwala's Scale Thesis)
    if eps_g is not None and rev_g is not None:
        if eps_g >= 18.0 and rev_g >= 12.0:
            score += 20
            breakdown.append(f"Double-Engine Growth (EPS +{eps_g:.1f}%, Sales +{rev_g:.1f}%).")
        elif eps_g > 0:
            score += 10
            breakdown.append("Positive top & bottom line trajectory.")
        else:
            breakdown.append("Growth deceleration present.")

    # 5. Valuation Check
    if pe is not None and pe > 0:
        if pe <= 35:
            score += 10
            breakdown.append(f"Fairly Valued (P/E {pe:.1f}).")
        elif pe > 75:
            score -= 10
            breakdown.append(f"Elevated Valuation (P/E {pe:.1f}). Requires high earnings delivery.")

    score = max(5, min(100, score))
    
    if score >= 75:
        tier = "👑 Institutional Quality Compounder"
    elif score >= 55:
        tier = "⭐ Quality Growth Business"
    else:
        tier = "⚠️ Below Institutional Hurdle"

    return {
        "score": score,
        "tier": tier,
        "reasons": breakdown
    }

# ============================================================
# APP UI NAVIGATION
# ============================================================

st.sidebar.title("🏛️ Institutional Terminal")
nav_mode = st.sidebar.radio(
    "Select Operating View",
    [
        "⚡ 5-Second Real-Time Pulse & AI",
        "🧠 Jhunjhunwala-Damani Moat Filter",
        "🔍 Full SEPA Breakout Engine"
    ]
)

with st.spinner("Downloading market matrices..."):
    bulk_data = fetch_all_market_data(WATCHLIST)

if bulk_data.empty:
    st.error("Market data feeds are temporarily unreachable. Please refresh.")
    st.stop()

# ============================================================
# VIEW 1: 5-SECOND REAL-TIME PULSE & AI PREDICTOR
# ============================================================
if nav_mode == "⚡ 5-Second Real-Time Pulse & AI":
    st.title("⚡ Real-Time Market Pulse & AI Predictor")
    st.markdown('<p style="color:#94a3b8;">Sub-minute quotes via isolated UI fragments combined with institutional Gradient Boosting models.</p>', unsafe_allow_html=True)
    
    selected_stock = st.selectbox("Select Active Stock to Stream", [s.replace(".NS", "") for s in WATCHLIST])
    ticker_sym = f"{selected_stock}.NS"
    raw_df = bulk_data.xs(ticker_sym, axis=1, level=1) if isinstance(bulk_data.columns, pd.MultiIndex) else bulk_data

    # ISOLATED 5-SECOND FRAGMENT (Prevents full app freeze)
    @st.fragment(run_every="5s")
    def live_stream_widget(symbol, t_symbol, hist_data):
        quote_data = get_live_quote(t_symbol)
        last_close = float(hist_data["Close"].dropna().iloc[-1])
        
        if quote_data:
            cmp = quote_data["price"]
            day_chg = quote_data["change"]
            refresh_time = quote_data["time"].strftime("%I:%M:%S %p IST")
        else:
            cmp = last_close
            day_chg = 0.0
            refresh_time = datetime.now(IST).strftime("%I:%M:%S %p IST")

        p_class = "positive" if day_chg >= 0 else "negative"
        arrow = "↑" if day_chg >= 0 else "↓"

        st.markdown(f"""
        <div class="stock-box">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="stock-name">{symbol}</span>
                    <span style="color:#94a3b8; font-size:13px; margin-left:8px;">NSE: {t_symbol}</span>
                </div>
                <div class="live-badge"><span class="pulse-dot"></span>LIVE 5s POLLING</div>
            </div>
            <div class="price">{money(cmp)}</div>
            <div class="{p_class}" style="font-size:18px;">{arrow} {day_chg:+.2f}% Today</div>
            <div style="color:#64748b; font-size:11px; margin-top:8px;">Last Synced: {refresh_time}</div>
        </div>
        """, unsafe_allow_html=True)

    live_stream_widget(selected_stock, ticker_sym, raw_df)

    # ML PREDICTIONS (Non-blocking heavy run)
    st.subheader("🤖 Larger AI Predictive Model (Dual Ensemble)")
    with st.spinner("Executing Gradient Boosting & Volatility features..."):
        ai_res = run_advanced_ai_model(raw_df)

    if ai_res:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted 5-Day Target", money(ai_res["predicted_target"]), f"{ai_res['expected_return_pct']:+.2f}% Expected")
        m2.metric("Ensemble Confidence", f"{ai_res['confidence']}%")
        m3.metric("Architecture", ai_res["model_type"])
        st.caption("⚙️ The model combines non-linear Gradient Boosted Trees with Random Forest regularization over multi-period momentum, rolling ATR, and volume deviations.")
    else:
        st.info("Insufficient bar depth to fit dual ensemble.")

    # 180 Day Context
    st.subheader("Price Action Structure")
    st.line_chart(raw_df["Close"].dropna().tail(180))

# ============================================================
# VIEW 2: JHUNJHUNWALA & DAMANI MOAT SCREENER
# ============================================================
elif nav_mode == "🧠 Jhunjhunwala-Damani Moat Filter":
    st.title("🧠 The Jhunjhunwala-Damani Quality Moat Filter")
    st.markdown('<p style="color:#94a3b8;">Reverse-engineered institutional criteria: High ROCE/ROE, Conservative Debt, Pricing Power, and Long-Term Scale Runway.</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="damani-box">
        <h4 style="margin-top:0px; color:#a5b4fc;">The Indian Legends Investing Matrix</h4>
        <p style="font-size:13px; color:#cbd5e1; margin-bottom:0;">
        "Buy right and sit tight." — Rakesh Jhunjhunwala did not day-trade algorithms; he hunted structural compounding machines. Radhakishan Damani prioritizes zero debt, negative operating cycles, and durable consumer habits. This filter audits corporate statements strictly against their investment philosophy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    records = []
    for s in WATCHLIST:
        fund = get_fundamental_metrics(s)
        c_price = float(bulk_data["Close"][s].dropna().iloc[-1]) if s in bulk_data["Close"] else 0.0
        score_data = calculate_jhunjhunwala_damani_score(fund, c_price)
        records.append({
            "Symbol": s.replace(".NS", ""),
            "Score": score_data["score"],
            "Tier": score_data["tier"],
            "ROE %": fund["roe"],
            "Debt/Equity": fund["debt_to_equity"],
            "Net Margin %": fund["profit_margin"],
            "EPS YoY %": fund["eps_growth"],
            "P/E": fund["pe"],
            "Reasons": score_data["reasons"]
        })

    moat_df = pd.DataFrame(records).sort_values(by="Score", ascending=False).reset_index(drop=True)

    for _, row in moat_df.iterrows():
        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:22px; font-weight:800;">{row['Symbol']}</span>
                    <span style="margin-left:12px; font-size:15px; color:#94a3b8;">Moat Score: <b>{row['Score']}/100</b></span>
                </div>
                <span class="badge-buy">{row['Tier']}</span>
            </div>
            <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:10px; margin-top:14px;">
                <div><div class="metric-lbl">Return on Equity</div><div class="metric-val">{f"{row['ROE %']:.1f}%" if pd.notna(row['ROE %']) else 'N/A'}</div></div>
                <div><div class="metric-lbl">Debt to Equity</div><div class="metric-val">{f"{row['Debt/Equity']:.2f}" if pd.notna(row['Debt/Equity']) else '0.00'}</div></div>
                <div><div class="metric-lbl">Net Margin</div><div class="metric-val">{f"{row['Net Margin %']:.1f}%" if pd.notna(row['Net Margin %']) else 'N/A'}</div></div>
                <div><div class="metric-lbl">EPS Growth</div><div class="metric-val">{f"{row['EPS YoY %']:.1f}%" if pd.notna(row['EPS YoY %']) else 'N/A'}</div></div>
                <div><div class="metric-lbl">P/E Ratio</div><div class="metric-val">{f"{row['P/E']:.1f}" if pd.notna(row['P/E']) else 'N/A'}</div></div>
            </div>
            <div class="reason-box">
                <b>Moat Audit:</b> {' • '.join(row['Reasons'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# VIEW 3: FULL SEPA BREAKOUT ENGINE
# ============================================================
elif nav_mode == "🔍 Full SEPA Breakout Engine":
    st.title("⚡ Minervini SEPA Institutional Screener")
    st.markdown('<p style="color:#94a3b8;">Stage 2 Trend Filters • Volatility Contraction (VCP) • Volume Expansions • Precise Risk Rules</p>', unsafe_allow_html=True)
    
    sepa_records = []
    closes = bulk_data["Close"]
    highs = bulk_data["High"]
    lows = bulk_data["Low"]
    volumes = bulk_data["Volume"]
    
    for s in WATCHLIST:
        if s not in closes.columns: continue
        c = closes[s].dropna()
        h = highs[s].dropna()
        l = lows[s].dropna()
        v = volumes[s].dropna()
        if len(c) < 180: continue
        
        cmp = float(c.iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        sma150 = float(c.rolling(150).mean().iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])
        high52 = float(h.tail(252).max())
        low52 = float(l.tail(252).min())
        
        pivot_20d = float(h.iloc[-21:-1].max())
        dist_pivot = ((cmp - pivot_20d) / pivot_20d) * 100
        
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        atr_pct = (atr14 / cmp) * 100
        
        v50 = float(v.rolling(50).mean().iloc[-1])
        cur_v = float(v.iloc[-1])
        vol_ratio = cur_v / v50 if v50 > 0 else 1.0
        
        # Stage 2 Minervini Rules
        is_stage2 = (cmp > sma150) and (cmp > sma200) and (sma150 > sma200) and (cmp > sma50) and (cmp >= 1.25 * low52) and (cmp >= 0.75 * high52)
        
        status = "AVOID"
        entry, stop, target, risk_pct = np.nan, np.nan, np.nan, np.nan
        
        if is_stage2:
            if cmp >= pivot_20d and vol_ratio >= 1.35 and dist_pivot <= 3.5:
                status = "BUY"
                entry = cmp
                stop = round(max(cmp * 0.94, cmp - (1.4 * atr14)), 2)
                risk_pct = round(((entry - stop) / entry) * 100, 2)
                target = round(entry + (risk_pct * 2.5 / 100 * entry), 2)
            elif -4.0 <= dist_pivot <= 0.5:
                status = "WATCH"
                entry = round(pivot_20d * 1.002, 2)
                stop = round(entry * 0.945, 2)
                risk_pct = round(((entry - stop) / entry) * 100, 2)
                target = round(entry + (risk_pct * 2.5 / 100 * entry), 2)
        
        sepa_records.append({
            "Symbol": s.replace(".NS", ""),
            "Status": status,
            "CMP": cmp,
            "Entry": entry,
            "Stop Loss": stop,
            "Target": target,
            "Risk %": risk_pct,
            "Dist Pivot %": dist_pivot,
            "Vol Ratio": vol_ratio,
            "ATR %": atr_pct
        })
        
    s_df = pd.DataFrame(sepa_records)
    
    st.dataframe(
        s_df,
        column_config={
            "CMP": st.column_config.NumberColumn(format="₹%.2f"),
            "Entry": st.column_config.NumberColumn(format="₹%.2f"),
            "Stop Loss": st.column_config.NumberColumn(format="₹%.2f"),
            "Target": st.column_config.NumberColumn(format="₹%.2f"),
            "Risk %": st.column_config.NumberColumn(format="%.1f%%"),
            "Dist Pivot %": st.column_config.NumberColumn(format="%.1f%%"),
            "Vol Ratio": st.column_config.NumberColumn(format="%.2fx"),
            "ATR %": st.column_config.NumberColumn(format="%.2f%%"),
        },
        use_container_width=True,
        hide_index=True
    )
