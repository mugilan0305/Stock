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
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG & UI
# ============================================================

st.set_page_config(page_title="Indian Stock AI Analyzer", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background: #0b1120; color: #f8fafc; }
    .block-container { max-width: 1400px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #f8fafc; }
    .subtitle { color: #94a3b8; margin-bottom: 20px; }
    .stock-box { background: #111827; border: 1px solid #263449; border-radius: 14px; padding: 22px; margin-bottom: 18px; }
    .stock-name { font-size: 30px; font-weight: 800; }
    .stock-symbol { color: #94a3b8; font-size: 13px; }
    .price { font-size: 40px; font-weight: 800; margin-top: 15px; }
    .positive { color: #22c55e; font-weight: 700; }
    .negative { color: #ef4444; font-weight: 700; }
    .section { font-size: 21px; font-weight: 800; margin-top: 28px; margin-bottom: 14px; }
    .reason { background: #111827; border: 1px solid #263449; border-radius: 12px; padding: 15px; color: #cbd5e1; line-height: 1.6; margin-bottom: 10px;}
    .gold-box { background: #3b2f10; border: 1px solid #b48600; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    </style>
    """,
    unsafe_allow_html=True
)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# ============================================================
# STOCK UNIVERSE (Top 100 NSE Liquid Stocks for Screener)
# ============================================================

NIFTY_100 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS", "ITC.NS", "SBIN.NS", "LT.NS", "HINDUNILVR.NS",
    "BAJFINANCE.NS", "AXISBANK.NS", "MARUTI.NS", "KOTAKBANK.NS", "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "TATASTEEL.NS", "NTPC.NS",
    "BAJAJFINSV.NS", "M&M.NS", "POWERGRID.NS", "TRENT.NS", "NESTLEIND.NS", "ADANIENT.NS", "TCS.NS", "HAL.NS", "ONGC.NS", "COALINDIA.NS",
    "ZOMATO.NS", "CHOLAFIN.NS", "HCLTECH.NS", "ADANIPORTS.NS", "BAJAJ-AUTO.NS", "TATASTEEL.NS", "INDUSINDBK.NS", "GRASIM.NS", "HINDALCO.NS", "TECHM.NS",
    "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "WIPRO.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "LTIM.NS", "HEROMOTOCO.NS", "TATAMOTORS.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "PIDILITIND.NS", "TATACONSUM.NS", "VBL.NS", "SHRIRAMFIN.NS", "TVSMOTOR.NS", "SIEMENS.NS", "DLF.NS", "BEL.NS",
    "INDIGO.NS", "JINDALSTEL.NS", "JSWSTEEL.NS", "PNB.NS", "BANKBARODA.NS", "IOC.NS", "BPCL.NS", "GAIL.NS", "HINDZINC.NS", "VEDL.NS",
    "AMBUJACEM.NS", "SHREECEM.NS", "BOSCHLTD.NS", "COLPAL.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "MCDOWELL-N.NS", "UBL.NS", "IRCTC.NS",
    "PFC.NS", "RECLTD.NS", "ICICIPRULI.NS", "ICICIGI.NS", "MUTHOOTFIN.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS", "AUROPHARMA.NS", "LUPIN.NS", "PIIND.NS"
]

NIFTY_10_DICT = {s.replace(".NS", ""): s for s in NIFTY_100[:10]}
NIFTY_50_DICT = {s.replace(".NS", ""): s for s in NIFTY_100[:50]}

def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except: return default

def money(value): return f"₹{safe_float(value):,.2f}"
def india_now(): return datetime.now(IST)

# ============================================================
# VECTORIZED SUPER SCREENER (MINERVINI TREND TEMPLATE)
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def minervini_screener(tickers):
    """
    Downloads all tickers at once in a vectorized format to bypass API limits.
    Applies Mark Minervini's Strict Trend Template for explosive growth stocks.
    """
    try:
        # Bulk download is incredibly fast for 100+ stocks
        df = yf.download(tickers, period="2y", auto_adjust=True, progress=False)
        if df.empty: return pd.DataFrame()

        closes = df['Close']
        highs = df['High']
        lows = df['Low']

        # Calculate Moving Averages for all stocks simultaneously
        sma50 = closes.rolling(50).mean()
        sma150 = closes.rolling(150).mean()
        sma200 = closes.rolling(200).mean()

        # 52 Week Highs and Lows (252 trading days)
        high_52w = highs.rolling(252).max()
        low_52w = lows.rolling(252).min()

        # Isolate the current day's values
        c_close = closes.iloc[-1]
        c_sma50 = sma50.iloc[-1]
        c_sma150 = sma150.iloc[-1]
        c_sma200 = sma200.iloc[-1]
        c_high52 = high_52w.iloc[-1]
        c_low52 = low_52w.iloc[-1]
        
        # Check if 200 SMA is trending up (compare to 20 days ago)
        sma200_20d_ago = sma200.iloc[-21]

        # ---------------------------------------------------------
        # MINERVINI'S 7 STRICT RULES FOR IDEAL INVESTMENTS
        # ---------------------------------------------------------
        # 1. Current Price is above both the 150-day and 200-day SMA
        cond1 = (c_close > c_sma150) & (c_close > c_sma200)
        # 2. The 150-day SMA is above the 200-day SMA
        cond2 = (c_sma150 > c_sma200)
        # 3. The 200-day SMA is trending up for at least 1 month
        cond3 = (c_sma200 > sma200_20d_ago)
        # 4. The 50-day SMA is above both 150 and 200 SMAs
        cond4 = (c_sma50 > c_sma150) & (c_sma50 > c_sma200)
        # 5. Current Price is above the 50-day SMA
        cond5 = (c_close > c_sma50)
        # 6. Current Price is at least 30% above its 52-week low
        cond6 = (c_close >= (1.3 * c_low52))
        # 7. Current Price is within 25% of its 52-week high
        cond7 = (c_close >= (0.75 * c_high52))

        # Combine all conditions (Only the best survive)
        passed_minervini = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7

        # Scoring Metric: Distance to 52 Week High (Lower % is better, implies strong momentum)
        distance_to_high = ((c_high52 - c_close) / c_high52) * 100

        results = pd.DataFrame({
            'Ticker': passed_minervini.index,
            'CMP': c_close.values,
            'Passed': passed_minervini.values,
            'Dist_to_High': distance_to_high.values,
            '52W High': c_high52.values,
            '50 SMA': c_sma50.values,
            '200 SMA': c_sma200.values
        })

        # Filter only stocks that passed, sort by closest to 52 week high breakout
        final_list = results[results['Passed'] == True].sort_values('Dist_to_High').reset_index(drop=True)
        return final_list

    except Exception as e:
        return pd.DataFrame()

# ============================================================
# STANDARD APP FUNCTIONS (Data, ML, News)
# ============================================================
# (Keeping your original core functions intact for the standard analyzer)

@st.cache_data(ttl=300, show_spinner=False)
def get_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(x in df.columns for x in required): return pd.DataFrame()
    return df[required].dropna()

def add_indicators(df):
    data = df.copy()
    close = data["Close"]
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    
    tr = pd.concat([data["High"] - data["Low"], (data["High"] - close.shift()).abs(), (data["Low"] - close.shift()).abs()], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(14).mean()
    data["RETURN5"] = close.pct_change(5)
    data["RETURN20"] = close.pct_change(20)
    return data

@st.cache_data(ttl=1800, show_spinner=False)
def predict_prices(data):
    if len(data) < 180: return None
    features = ["Close", "SMA20", "SMA50", "SMA200", "RSI", "MACD", "MACD_SIGNAL", "ATR", "RETURN5", "RETURN20"]
    df = data.copy()
    df["TARGET1"] = df["Close"].shift(-1)
    df["TARGET5"] = df["Close"].shift(-5)
    clean = df.dropna(subset=features + ["TARGET1", "TARGET5"])
    if len(clean) < 100: return None
    latest = df[features].iloc[[-1]].dropna()
    if latest.empty: return None

    predictions = {}
    for name, target, seed in [("1d", "TARGET1", 42), ("5d", "TARGET5", 43)]:
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=seed, n_jobs=-1)
        model.fit(clean[features], clean[target])
        predictions[name] = safe_float(model.predict(latest)[0])
    return predictions

# ============================================================
# SIDEBAR UI
# ============================================================

st.sidebar.title("📊 AI Dashboard")

app_mode = st.sidebar.radio("Select Mode:", ["Single Stock Analyzer", "👑 Minervini Super Screener"])

if app_mode == "Single Stock Analyzer":
    universe_name = st.sidebar.selectbox("Universe", ["Nifty 10", "Nifty 50", "Custom"])
    if universe_name == "Nifty 10":
        universe = NIFTY_10_DICT
        symbol = st.sidebar.selectbox("Select Stock", list(universe.keys()))
        ticker = universe[symbol]
    elif universe_name == "Nifty 50":
        universe = NIFTY_50_DICT
        symbol = st.sidebar.selectbox("Select Stock", list(universe.keys()))
        ticker = universe[symbol]
    else:
        symbol = st.sidebar.text_input("NSE Symbol", "RELIANCE").upper().strip()
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ============================================================
# MAIN APP LOGIC
# ============================================================

if app_mode == "👑 Minervini Super Screener":
    st.title("👑 The Ideal Investment Screener")
    st.markdown('<div class="subtitle">Vectorized scanning of Top 100 NSE stocks using Mark Minervini\'s strict Trend Template.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="gold-box">
        <h3>What makes an "Ideal" stock?</h3>
        <p>Short-term AI predictions are for trading. For true <b>investing</b>, we look for massive institutional accumulation. 
        This screener identifies stocks that meet all 7 of Mark Minervini's unshakeable rules simultaneously:</p>
        <ol>
            <li>Current Price is strictly above 150 & 200 Day Moving Averages.</li>
            <li>150 DMA is strictly above the 200 DMA.</li>
            <li>200 DMA has been trending upward for at least 1 month.</li>
            <li>50 DMA is strictly above both the 150 and 200 DMAs.</li>
            <li>Current Price is strictly above the 50 DMA.</li>
            <li>Price is at least 30% above its 52-week low.</li>
            <li>Price is within 25% of its 52-week high.</li>
        </ol>
        <p><i>The stocks that survive this filter are ranked by their proximity to breaking out of their 52-week high.</i></p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Downloading 100+ stocks and running vector math... (Takes ~3 seconds)"):
        top_stocks = minervini_screener(NIFTY_100)
    
    if not top_stocks.empty:
        st.success(f"🔥 Found {len(top_stocks)} stocks out of 100 currently in absolute Stage 2 Uptrends!")
        
        # Clean up dataframe for display
        display_df = top_stocks.copy()
        display_df['Ticker'] = display_df['Ticker'].str.replace('.NS', '')
        display_df['CMP'] = display_df['CMP'].apply(lambda x: f"₹{x:.2f}")
        display_df['52W High'] = display_df['52W High'].apply(lambda x: f"₹{x:.2f}")
        display_df['Dist_to_High'] = display_df['Dist_to_High'].apply(lambda x: f"{x:.1f}% below high")
        display_df['50 SMA'] = display_df['50 SMA'].apply(lambda x: f"₹{x:.2f}")
        display_df['200 SMA'] = display_df['200 SMA'].apply(lambda x: f"₹{x:.2f}")
        
        display_df = display_df.rename(columns={'Dist_to_High': 'Breakout Proximity', 'CMP': 'Current Price'})
        display_df = display_df.drop(columns=['Passed'])
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Next Steps")
        st.info("Pick the top 2-3 stocks from the list above. Switch to the 'Single Stock Analyzer' mode on the left, type in their ticker, and look at the short-term AI Trade Plan to find the safest entry point with a tight Stop Loss.")
    else:
        st.error("Market is weak! ZERO stocks out of the Nifty 100 currently meet the strict Minervini Trend Template. Cash is the safest position.")

elif app_mode == "Single Stock Analyzer":
    st.title("📈 Indian Stock AI Analyzer")
    
    with st.spinner(f"Analyzing {symbol}..."):
        data = get_data(ticker)
        
    if data.empty:
        st.error("Unable to retrieve data for this stock.")
        st.stop()
        
    data = add_indicators(data)
    last = data.iloc[-1]
    current = safe_float(last["Close"])
    previous = safe_float(data["Close"].iloc[-2])
    today_change = ((current / previous - 1) * 100 if previous else 0)
    
    price_class = "positive" if today_change >= 0 else "negative"
    arrow = "↑" if today_change >= 0 else "↓"
    
    st.markdown(f"""
        <div class="stock-box">
            <div class="stock-name">{html.escape(symbol)}</div>
            <div class="stock-symbol">NSE • {html.escape(ticker)}</div>
            <div class="price">{money(current)}</div>
            <div class="{price_class}">{arrow} {abs(today_change):.2f}% today</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("🤖 Short Term AI Trade Plan")
    ml = predict_prices(data)
    
    if ml:
        t1, t2, t3 = st.columns(3)
        t1.metric("Current Price", money(current))
        t2.metric("AI Target (1 Day)", money(ml['1d']), f"{((ml['1d']/current)-1)*100:.2f}%")
        t3.metric("AI Target (5 Day)", money(ml['5d']), f"{((ml['5d']/current)-1)*100:.2f}%")
        
        atr = safe_float(last["ATR"])
        stop_loss = current - (atr * 1.2)
        st.warning(f"**Suggested Stop Loss:** {money(stop_loss)} (Placed securely below current volatility limits)")
    else:
        st.info("Not enough data to run ML predictions.")

    st.subheader("📊 Technical Indicators")
    a, b, c, d = st.columns(4)
    a.metric("RSI", f'{safe_float(last["RSI"]):.1f}')
    b.metric("SMA 50", money(last["SMA50"]))
    c.metric("SMA 200", money(last["SMA200"]))
    d.metric("MACD", f'{safe_float(last["MACD"]):.2f}')
    
    st.subheader("📈 Price Trend")
    st.line_chart(data[["Close", "SMA50", "SMA200"]].tail(180))
