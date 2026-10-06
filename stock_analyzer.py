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
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# APP CONFIG & DARK THEME
# ============================================================

st.set_page_config(
    page_title="Institutional Quantitative Terminal",
    page_icon="🏛️️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background: #0b1120; color: #f8fafc; }
.block-container { max-width: 1440px; padding-top: 1.2rem; padding-bottom: 3rem; }
.card { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.badge-buy { background: #064e3b; color: #34d399; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; text-align: center; }
.badge-watch { background: #451a03; color: #fbbf24; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; text-align: center; }
.badge-avoid { background: #27272a; color: #a1a1aa; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; text-align: center; }
.badge-sell { background: #450a0a; color: #f87171; font-weight: 800; padding: 6px 12px; border-radius: 6px; font-size: 13px; text-align: center; }
.metric-val { font-size: 20px; font-weight: 800; color: #f8fafc; margin-top: 4px; }
.metric-lbl { font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
.live-badge { display: inline-flex; align-items: center; background: #0f2e1b; border: 1px solid #10b981; color: #34d399; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 12px; }
.pulse-dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; display: inline-block; margin-right: 6px; }
.stock-box { background: #111827; border: 1px solid #1e293b; border-radius: 14px; padding: 22px; margin-bottom: 18px; }
.price { font-size: 38px; font-weight: 800; margin-top: 8px; }
.positive { color: #22c55e; font-weight: 700; }
.negative { color: #ef4444; font-weight: 700; }
.top-pick-card { background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%); border: 2px solid #818cf8; border-radius: 14px; padding: 22px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

IST = ZoneInfo("Asia/Kolkata")

WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS",
    "ITC.NS", "SBIN.NS", "LT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "AXISBANK.NS",
    "MARUTI.NS", "KOTAKBANK.NS", "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "TATASTEEL.NS", "NTPC.NS", "M&M.NS", "POWERGRID.NS", "TRENT.NS", "HAL.NS", "BEL.NS"
]

# ============================================================
# DATA PIPELINE
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
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        eps_g = info.get("earningsGrowth", None)
        rev_g = info.get("revenueGrowth", None)
        roe = info.get("returnOnEquity", None)
        de = info.get("debtToEquity", None)
        pe = info.get("trailingPE", None)
        pm = info.get("profitMargins", None)
        
        peg = None
        if pe is not None and eps_g is not None and eps_g > 0:
            peg = pe / (eps_g * 100)

        return {
            "eps_growth": (eps_g * 100) if eps_g is not None and np.isfinite(eps_g) else None,
            "rev_growth": (rev_g * 100) if rev_g is not None and np.isfinite(rev_g) else None,
            "roe": (roe * 100) if roe is not None and np.isfinite(roe) else None,
            "debt_to_equity": (de / 100) if de is not None and np.isfinite(de) else None,
            "pe": pe if pe is not None and np.isfinite(pe) else None,
            "peg": peg if peg is not None and np.isfinite(peg) else None,
            "profit_margin": (pm * 100) if pm is not None and np.isfinite(pm) else None
        }
    except Exception:
        return {"eps_growth": None, "rev_growth": None, "roe": None, "debt_to_equity": None, "pe": None, "peg": None, "profit_margin": None}

def get_live_quote(ticker):
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
# INSTITUTIONAL STRATEGY MODELS
# ============================================================

def evaluate_institutional_models(fund_data, tech_data):
    roe = fund_data.get("roe") or 0.0
    de = fund_data.get("debt_to_equity") or 0.0
    margin = fund_data.get("profit_margin") or 0.0
    eps_g = fund_data.get("eps_growth") or 0.0
    rev_g = fund_data.get("rev_growth") or 0.0
    pe = fund_data.get("pe") or 0.0
    peg = fund_data.get("peg") or 99.0
    is_stage2 = tech_data.get("is_stage2", False)

    # 1. Growth & Scale
    s_growth = 0
    if roe >= 20: s_growth += 35
    elif roe >= 15: s_growth += 20
    if eps_g >= 18: s_growth += 35
    elif eps_g >= 10: s_growth += 20
    if rev_g >= 15: s_growth += 20
    if is_stage2: s_growth += 10

    # 2. Fortress Balance Sheet
    s_moat = 0
    if de <= 0.2: s_moat += 40
    elif de <= 0.5: s_moat += 25
    elif de > 1.0: s_moat -= 20
    if margin >= 18: s_moat += 35
    elif margin >= 12: s_moat += 20
    if roe >= 18: s_moat += 25

    # 3. GARP (Growth at a Reasonable Price)
    s_garp = 0
    if roe >= 20: s_garp += 30
    if eps_g >= 15 and rev_g >= 12: s_garp += 30
    if 0 < peg <= 1.5: s_garp += 30
    elif 1.5 < peg <= 2.2: s_garp += 15
    if is_stage2: s_garp += 10

    # 4. Momentum & Acceleration
    s_mom = 0
    if eps_g >= 22: s_mom += 40
    elif eps_g >= 12: s_mom += 20
    if de <= 0.6: s_mom += 25
    if is_stage2: s_mom += 35

    # 5. Deep Value
    s_value = 0
    if 0 < pe <= 25: s_value += 40
    elif 25 < pe <= 35: s_value += 20
    if de <= 0.3: s_value += 30
    if roe >= 16: s_value += 20
    if eps_g > 0: s_value += 10

    s_growth = max(0, min(100, s_growth))
    s_moat = max(0, min(100, s_moat))
    s_garp = max(0, min(100, s_garp))
    s_mom = max(0, min(100, s_mom))
    s_value = max(0, min(100, s_value))

    composite = round((0.25 * s_growth) + (0.25 * s_moat) + (0.20 * s_garp) + (0.15 * s_mom) + (0.15 * s_value), 1)

    return {
        "Composite": composite,
        "Growth_Scale": s_growth,
        "Fortress_Moat": s_moat,
        "GARP": s_garp,
        "Momentum": s_mom,
        "Deep_Value": s_value
    }

def get_signal_badge(signal_text):
    if "BUY" in signal_text.upper():
        return f'<span class="badge-buy">{signal_text}</span>'
    elif "SELL" in signal_text.upper():
        return f'<span class="badge-sell">{signal_text}</span>'
    elif "WATCH" in signal_text.upper():
        return f'<span class="badge-watch">{signal_text}</span>'
    else:
        return f'<span class="badge-avoid">{signal_text}</span>'

# ============================================================
# APP UI & NAVIGATION
# ============================================================

st.sidebar.title("🏛️ Terminal Navigation")

st.sidebar.markdown("""
<div style="background:#0f2e1b; border:1px solid #10b981; border-radius:8px; padding:12px; text-align:center; margin-bottom:20px;">
    <div style="font-size:11px; color:#34d399; font-weight:700; margin-bottom:4px; letter-spacing:1px;">LIVE MARKET CLOCK (IST)</div>
    <div id="live-clock" style="font-size:18px; font-family:monospace; color:#f8fafc; font-weight:bold;">Loading...</div>
</div>
<script>
    setInterval(() => {
        let options = { timeZone: 'Asia/Kolkata', hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' };
        document.getElementById('live-clock').innerText = new Date().toLocaleTimeString('en-IN', options);
    }, 1000);
</script>
""", unsafe_allow_html=True)

nav_mode = st.sidebar.radio(
    "Select Operating View",
    [
        "🏆 Institutional Consensus Matrix",
        "⚡ 5-Second Real-Time Pulse & AI",
        "🔍 Single Stock Target Diagnosis"
    ]
)

with st.spinner("Downloading market matrices..."):
    bulk_data = fetch_all_market_data(WATCHLIST)

if bulk_data.empty:
    st.error("Market data feeds are temporarily unreachable. Please refresh.")
    st.stop()

# ============================================================
# VIEW 1: INSTITUTIONAL CONSENSUS MATRIX
# ============================================================
if nav_mode == "🏆 Institutional Consensus Matrix":
    st.title("🏆 Institutional Consensus Matrix")
    st.markdown('<p style="color:#94a3b8;">Audits Indian large caps across 5 core quantitative methodologies. Includes explicit Buy/Sell/Avoid signals based on technical alignment and institutional scoring.</p>', unsafe_allow_html=True)

    closes = bulk_data["Close"]
    highs = bulk_data["High"]
    lows = bulk_data["Low"]

    matrix_records = []
    
    for s in WATCHLIST:
        if s not in closes.columns: continue
        c = closes[s].dropna()
        h = highs[s].dropna()
        l = lows[s].dropna()
        if len(c) < 180: continue
        
        cmp = float(c.iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        sma150 = float(c.rolling(150).mean().iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])
        high52 = float(h.tail(252).max())
        low52 = float(l.tail(252).min())
        
        is_stage2 = (cmp > sma150) and (cmp > sma200) and (sma150 > sma200) and (cmp > sma50) and (cmp >= 1.25 * low52) and (cmp >= 0.75 * high52)
        
        pivot_20d = float(h.iloc[-21:-1].max())
        atr14 = float((h - l).rolling(14).mean().iloc[-1])
        
        fund = get_fundamental_metrics(s)
        scores = evaluate_institutional_models(fund, {"is_stage2": is_stage2})
        
        stop_loss = round(max(cmp * 0.94, cmp - (1.4 * atr14)), 2)
        risk_pct = round(((cmp - stop_loss) / cmp) * 100, 2)
        target = round(cmp + (risk_pct * 2.5 / 100 * cmp), 2)

        # Generate Explicit Signal
        if is_stage2 and scores["Composite"] >= 65:
            signal_out = "BUY"
        elif cmp < sma200 or scores["Composite"] < 40:
            signal_out = "SELL"
        elif not is_stage2:
            signal_out = "AVOID"
        else:
            signal_out = "WATCH"

        matrix_records.append({
            "Symbol": s.replace(".NS", ""),
            "Signal": signal_out,
            "CMP": cmp,
            "Consensus Score": scores["Composite"],
            "Growth & Scale": scores["Growth_Scale"],
            "Fortress Moat": scores["Fortress_Moat"],
            "GARP": scores["GARP"],
            "Momentum": scores["Momentum"],
            "Deep Value": scores["Deep_Value"],
            "ROE %": fund["roe"],
            "D/E": fund["debt_to_equity"],
            "P/E": fund["pe"]
        })

    consensus_df = pd.DataFrame(matrix_records).sort_values(by="Consensus Score", ascending=False).reset_index(drop=True)

    if not consensus_df.empty:
        top_pick = consensus_df.iloc[0]
        
        st.markdown(f"""
        <div class="top-pick-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:12px; color:#a5b4fc; text-transform:uppercase; font-weight:700; letter-spacing:1px;">🏆 Consensus Top Institutional Candidate</span>
                    <div style="font-size:32px; font-weight:900; color:#ffffff; margin-top:2px;">{top_pick['Symbol']}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:28px; font-weight:900; color:#38bdf8;">{top_pick['Consensus Score']} / 100</div>
                    {get_signal_badge(top_pick['Signal'])}
                </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-top:18px;">
                <div><div class="metric-lbl">Current CMP</div><div class="metric-val">{money(top_pick['CMP'])}</div></div>
                <div><div class="metric-lbl">ROE %</div><div class="metric-val">{top_pick['ROE %']:.1f}%</div></div>
                <div><div class="metric-lbl">Debt / Equity</div><div class="metric-val">{top_pick['D/E']:.2f}</div></div>
                <div><div class="metric-lbl">P/E Ratio</div><div class="metric-val">{top_pick['P/E']:.1f}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Cross-Strategy Comparison Grid")
    
    # Render table with styling for Signals
    def highlight_signal(val):
        if val == 'BUY': return 'color: #34d399; font-weight: bold'
        elif val == 'SELL': return 'color: #f87171; font-weight: bold'
        elif val == 'WATCH': return 'color: #fbbf24; font-weight: bold'
        return 'color: #a1a1aa'
        
    styled_df = consensus_df[["Symbol", "Signal", "CMP", "Consensus Score", "Growth & Scale", "Fortress Moat", "GARP", "Momentum", "Deep Value", "ROE %", "D/E", "P/E"]]
    
    st.dataframe(
        styled_df.style.map(highlight_signal, subset=['Signal']),
        column_config={
            "CMP": st.column_config.NumberColumn(format="₹%.2f"),
            "Consensus Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
            "ROE %": st.column_config.NumberColumn(format="%.1f%%"),
            "D/E": st.column_config.NumberColumn(format="%.2f"),
            "P/E": st.column_config.NumberColumn(format="%.1f"),
        },
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# VIEW 2: 5-SECOND REAL-TIME PULSE & AI
# ============================================================
elif nav_mode == "⚡ 5-Second Real-Time Pulse & AI":
    st.title("⚡ Real-Time Market Pulse & AI Predictor")
    st.markdown('<p style="color:#94a3b8;">Sub-minute quotes via isolated UI fragments combined with quantitative Gradient Boosting models.</p>', unsafe_allow_html=True)
    
    selected_stock = st.selectbox("Select Active Stock to Stream", [s.replace(".NS", "") for s in WATCHLIST])
    ticker_sym = f"{selected_stock}.NS"
    raw_df = bulk_data.xs(ticker_sym, axis=1, level=1) if isinstance(bulk_data.columns, pd.MultiIndex) else bulk_data

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
            <div style="color:#64748b; font-size:11px; margin-top:8px;">Last Synced Data: {refresh_time}</div>
        </div>
        """, unsafe_allow_html=True)

    live_stream_widget(selected_stock, ticker_sym, raw_df)

    @st.cache_data(ttl=900, show_spinner=False)
    def run_advanced_ai_model(df_stock):
        if len(df_stock) < 180: return None
        df = df_stock.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"]

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

        vol_mean = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std().replace(0,
