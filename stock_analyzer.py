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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian Stock AI Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIMPLE DARK UI
# ============================================================

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
    .neutral { color: #f59e0b; font-weight: 700; }
    .section { font-size: 21px; font-weight: 800; margin-top: 28px; margin-bottom: 14px; }
    .reason { background: #111827; border: 1px solid #263449; border-radius: 12px; padding: 15px; color: #cbd5e1; line-height: 1.6; margin-bottom: 10px;}
    .news { background: #111827; border: 1px solid #263449; border-radius: 10px; padding: 12px; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CONSTANTS
# ============================================================

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# ============================================================
# STOCK LISTS
# ============================================================

NIFTY_10 = {
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS", "ICICIBANK": "ICICIBANK.NS", "BHARTIARTL": "BHARTIARTL.NS",
    "ITC": "ITC.NS", "SBIN": "SBIN.NS", "LT": "LT.NS", "AXISBANK": "AXISBANK.NS",
}

NIFTY_50 = {
    "ADANIENT": "ADANIENT.NS", "ADANIPORTS": "ADANIPORTS.NS", "APOLLOHOSP": "APOLLOHOSP.NS",
    "ASIANPAINT": "ASIANPAINT.NS", "AXISBANK": "AXISBANK.NS", "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "BAJFINANCE": "BAJFINANCE.NS", "BAJAJFINSV": "BAJAJFINSV.NS", "BEL": "BEL.NS",
    "BHARTIARTL": "BHARTIARTL.NS", "BPCL": "BPCL.NS", "BRITANNIA": "BRITANNIA.NS",
    "CIPLA": "CIPLA.NS", "COALINDIA": "COALINDIA.NS", "DRREDDY": "DRREDDY.NS",
    "EICHERMOT": "EICHERMOT.NS", "GRASIM": "GRASIM.NS", "HCLTECH": "HCLTECH.NS",
    "HDFCBANK": "HDFCBANK.NS", "HDFCLIFE": "HDFCLIFE.NS", "HEROMOTOCO": "HEROMOTOCO.NS",
    "HINDALCO": "HINDALCO.NS", "HINDUNILVR": "HINDUNILVR.NS", "ICICIBANK": "ICICIBANK.NS",
    "INDUSINDBK": "INDUSINDBK.NS", "INFY": "INFY.NS", "ITC": "ITC.NS",
    "JIOFIN": "JIOFIN.NS", "JSWSTEEL": "JSWSTEEL.NS", "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS", "M&M": "M&M.NS", "MARUTI": "MARUTI.NS", "MAXHEALTH": "MAXHEALTH.NS",
    "NESTLEIND": "NESTLEIND.NS", "NTPC": "NTPC.NS", "ONGC": "ONGC.NS",
    "POWERGRID": "POWERGRID.NS", "RELIANCE": "RELIANCE.NS", "SBILIFE": "SBILIFE.NS",
    "SBIN": "SBIN.NS", "SHRIRAMFIN": "SHRIRAMFIN.NS", "SUNPHARMA": "SUNPHARMA.NS",
    "TATACONSUM": "TATACONSUM.NS", "TATAMOTORS": "TATAMOTORS.NS", "TATASTEEL": "TATASTEEL.NS",
    "TCS": "TCS.NS", "TECHM": "TECHM.NS", "TITAN": "TITAN.NS",
    "TRENT": "TRENT.NS", "ULTRACEMCO": "ULTRACEMCO.NS",
}

# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except Exception:
        return default

def money(value):
    return f"₹{safe_float(value):,.2f}"

def india_now():
    return datetime.now(IST)

def market_status():
    now = india_now()
    if now.weekday() >= 5:
        return "Market Closed", "Weekend"
    if now.time() < MARKET_OPEN:
        return "Market Closed", "Pre-market"
    if now.time() < MARKET_CLOSE:
        return "Market Open", "Live market"
    return "Market Closed", "After market"

# ============================================================
# LIVE PRICE
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def get_live_price(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker)}"
        params = {"interval": "1m", "range": "1d", "includePrePost": "false"}
        response = requests.get(url, params=params, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        result = response.json()["chart"]["result"][0]
        meta = result.get("meta", {})
        
        price = safe_float(meta.get("regularMarketPrice"), np.nan)
        previous_close = safe_float(meta.get("previousClose"), np.nan)
        timestamp = meta.get("regularMarketTime")
        
        quote_time = None
        if timestamp:
            quote_time = datetime.fromtimestamp(int(timestamp), tz=IST)
            
        if not np.isfinite(price):
            timestamps = result.get("timestamp", [])
            quotes = result.get("indicators", {}).get("quote", [])
            closes = quotes[0].get("close", []) if quotes else []
            
            for ts, close in reversed(list(zip(timestamps, closes))):
                value = safe_float(close, np.nan)
                if np.isfinite(value):
                    price = value
                    quote_time = datetime.fromtimestamp(int(ts), tz=IST)
                    break
                    
        if not np.isfinite(price):
            return None
            
        if not np.isfinite(previous_close):
            previous_close = price
            
        change = ((price / previous_close - 1) * 100)
        
        return {
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "quote_time": quote_time,
        }
    except Exception:
        return None

# ============================================================
# HISTORICAL DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [str(c).title() for c in df.columns]
        required = ["Open", "High", "Low", "Close", "Volume"]
        
        if not all(x in df.columns for x in required):
            return pd.DataFrame()
            
        return df[required].dropna()
    except Exception:
        return pd.DataFrame()

# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # Moving Averages
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(14).mean()

    # NEW: Bollinger Bands
    data["BB_MID"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data["BB_UPPER"] = data["BB_MID"] + (bb_std * 2)
    data["BB_LOWER"] = data["BB_MID"] - (bb_std * 2)
    
    # NEW: Stochastic Oscillator
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    data["STOCH_K"] = 100 * ((close - low_14) / (high_14 - low_14))
    data["STOCH_D"] = data["STOCH_K"].rolling(3).mean()

    data["VOL20"] = volume.rolling(20).mean()
    data["RETURN5"] = close.pct_change(5)
    data["RETURN20"] = close.pct_change(20)

    return data

# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(data):
    last = data.iloc[-1]
    close = safe_float(last["Close"])
    sma20 = safe_float(last["SMA20"])
    sma50 = safe_float(last["SMA50"])
    sma200 = safe_float(last["SMA200"])
    rsi = safe_float(last["RSI"], 50)
    macd = safe_float(last["MACD"])
    signal = safe_float(last["MACD_SIGNAL"])

    score = 0
    score += 2 if close > sma20 else -2
    score += 2 if close > sma50 else -2
    score += 2 if sma50 > sma200 else -2
    score += 2 if macd > signal else -2

    if 50 <= rsi <= 70:
        score += 1
    elif rsi < 30:
        score += 1
    elif rsi > 75:
        score -= 1

    return score

# ============================================================
# AI PREDICTION (Optimized)
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def predict_prices(data, fast_mode=False):
    if len(data) < 180:
        return None

    features = [
        "Close", "SMA20", "SMA50", "SMA200", "RSI", 
        "MACD", "MACD_SIGNAL", "ATR", "RETURN5", "RETURN20",
        "BB_UPPER", "BB_LOWER", "STOCH_K", "STOCH_D"
    ]

    df = data.copy()
    df["TARGET1"] = df["Close"].shift(-1)
    df["TARGET5"] = df["Close"].shift(-5)
    df["TARGET20"] = df["Close"].shift(-20)

    clean = df.dropna(subset=features + ["TARGET1", "TARGET5", "TARGET20"])
    if len(clean) < 100:
        return None

    latest = df[features].iloc[[-1]].dropna()
    if latest.empty:
        return None

    predictions = {}
    
    # Fast mode uses fewer trees to scan 50 stocks rapidly
    n_trees = 20 if fast_mode else 150
    max_d = 5 if fast_mode else 8

    for name, target, seed in [
        ("1d", "TARGET1", 42),
        ("5d", "TARGET5", 43),
        ("20d", "TARGET20", 44),
    ]:
        model = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_d,
            random_state=seed,
            n_jobs=-1
        )
        
        model.fit(clean[features], clean[target])
        predictions[name] = safe_float(model.predict(latest)[0])

    return predictions

# ============================================================
# MARKET TREND
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_market_trend():
    data = get_data("^NSEI")
    if data.empty:
        return "Unknown", 0
        
    data = add_indicators(data)
    last = data.iloc[-1]
    
    close = safe_float(last["Close"])
    sma20 = safe_float(last["SMA20"])
    sma50 = safe_float(last["SMA50"])
    
    score = 0
    score += 1 if close > sma20 else -1
    score += 1 if close > sma50 else -1
    
    if score == 2: return "Bullish", 2
    if score == -2: return "Bearish", -2
    return "Neutral", score

# ============================================================
# NEWS
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol):
    try:
        query = quote(f"{symbol} India stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        
        if response.status_code != 200:
            return []
            
        root = ET.fromstring(response.content)
        articles = []
        
        for item in root.findall(".//item")[:8]:
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

def news_sentiment(news):
    positive_words = ["profit", "growth", "strong", "upgrade", "buy", "bullish", "surge", "rally", "positive", "record", "order", "deal", "contract"]
    negative_words = ["loss", "weak", "downgrade", "sell", "bearish", "fall", "drop", "decline", "negative", "debt", "risk", "warning"]
    
    score = 0
    for article in news:
        title = article["title"].lower()
        for word in positive_words:
            if word in title: score += 1
        for word in negative_words:
            if word in title: score -= 1
            
    if score >= 3: sentiment = "Positive"
    elif score <= -3: sentiment = "Negative"
    else: sentiment = "Neutral"
    
    return score, sentiment

# ============================================================
# SIGNAL & TRADE LEVELS
# ============================================================

def generate_signal(technical, news_score, prediction_change, market_score):
    score = (
        technical * 0.55 + 
        news_score * 0.20 + 
        np.sign(prediction_change) * min(abs(prediction_change), 5) * 0.25 + 
        market_score * 0.50
    )
    if score >= 5: return "STRONG BUY"
    if score >= 2: return "BUY"
    if score <= -5: return "STRONG SELL"
    if score <= -2: return "SELL"
    return "HOLD"

def trade_levels(current, prediction, atr, signal):
    current = safe_float(current)
    prediction = safe_float(prediction)
    atr = safe_float(atr)
    if atr <= 0: atr = current * 0.02

    if signal in ["BUY", "STRONG BUY"]:
        target1 = max(prediction, current + atr * 0.75)
        target2 = max(current + atr * 1.50, prediction * 1.015)
        stop = current - atr * 1.20
        risk = max(current - stop, 0.01)
        reward = max(target1 - current, 0)
    elif signal in ["SELL", "STRONG SELL"]:
        target1 = min(prediction, current - atr * 0.75)
        target2 = min(current - atr * 1.50, prediction * 0.985)
        stop = current + atr * 1.20
        risk = max(stop - current, 0.01)
        reward = max(current - target1, 0)
    else:
        target1 = prediction
        target2 = prediction
        stop = current - atr if prediction >= current else current + atr
        risk = abs(current - stop)
        reward = abs(target1 - current)

    rr = reward / risk if risk > 0 else 0
    return {
        "target1": max(target1, 0.01),
        "target2": max(target2, 0.01),
        "stop": max(stop, 0.01),
        "rr": rr,
        "atr_used": atr
    }

# ============================================================
# ANALYZE STOCK
# ============================================================

def analyze_stock(symbol, ticker, market_name, market_score, fast_mode=False):
    data = get_data(ticker)
    if data.empty:
        return None
        
    data = add_indicators(data)
    data = data.dropna(subset=["Close"])
    if len(data) < 60:
        return None
        
    last = data.iloc[-1]
    model_close = safe_float(last["Close"])
    
    live = get_live_price(ticker)
    if live:
        current = live["price"]
        today_change = live["change"]
    else:
        current = model_close
        previous = safe_float(data["Close"].iloc[-2])
        today_change = ((current / previous - 1) * 100 if previous else 0)

    tech = technical_score(data)
    news = get_news(symbol)
    news_score, sentiment = news_sentiment(news)
    
    # ML with fast_mode parameter to optimize scanner
    ml = predict_prices(data, fast_mode=fast_mode)
    if ml:
        model_changes = {
            "1d": ((ml["1d"] / model_close - 1) * 100),
            "5d": ((ml["5d"] / model_close - 1) * 100),
            "20d": ((ml["20d"] / model_close - 1) * 100),
        }
    else:
        model_changes = {"1d": 0.5, "5d": 1.5, "20d": 3.0}

    predictions = {
        "1d": current * (1 + model_changes["1d"] / 100),
        "5d": current * (1 + model_changes["5d"] / 100),
        "20d": current * (1 + model_changes["20d"] / 100),
    }

    signal = generate_signal(tech, news_score, model_changes["1d"], market_score)
    levels = trade_levels(current, predictions["1d"], safe_float(last["ATR"]), signal)

    confidence = 50
    confidence += min(abs(tech) * 4, 20)
    confidence += min(abs(news_score) * 2, 10)
    confidence += (abs(market_score) * 4)
    if levels["rr"] >= 2: confidence += 10
    confidence += min(abs(model_changes["1d"]), 10)
    confidence = max(25, min(confidence, 95))

    return {
        "data": data, "last": last, "current": current, "today_change": today_change,
        "signal": signal, "tech": tech, "news": news, "news_score": news_score,
        "sentiment": sentiment, "predictions": predictions, "changes": model_changes,
        "levels": levels, "confidence": confidence, "live": live,
    }


# ============================================================
# SIDEBAR UI
# ============================================================

st.sidebar.title("📊 Stock Analyzer")

universe_name = st.sidebar.selectbox(
    "Stock Universe",
    ["Nifty 10", "Nifty 50", "Custom Stock", "My Portfolio"]
)

if universe_name == "Nifty 10":
    universe = NIFTY_10
elif universe_name == "Nifty 50":
    universe = NIFTY_50
elif universe_name == "My Portfolio":
    portfolio_input = st.sidebar.text_input("Enter symbols (comma separated)", "RELIANCE, TCS, INFY")
    universe = {}
    for s in portfolio_input.split(","):
        clean_s = s.strip().upper()
        if clean_s:
            ticker = clean_s if clean_s.endswith(".NS") else f"{clean_s}.NS"
            universe[clean_s] = ticker
else:
    universe = {}

if universe_name == "Custom Stock":
    symbol = st.sidebar.text_input("NSE Symbol", "RELIANCE").upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
else:
    symbol = st.sidebar.selectbox("Select Stock", list(universe.keys()))
    ticker = universe[symbol]

scan = st.sidebar.checkbox("Scan entire universe")

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ============================================================
# MAIN HEADER
# ============================================================

st.title("📈 Indian Stock AI Analyzer")
st.markdown('<div class="subtitle">Simple technical analysis • AI prediction • Trading signals</div>', unsafe_allow_html=True)

market_name, market_score = get_market_trend()

# ============================================================
# SCANNER VIEW
# ============================================================

if scan and len(universe) > 1:
    st.subheader(f"🔎 {universe_name} Scanner")
    rows = []
    progress = st.progress(0)
    stocks = list(universe.items())

    for i, (stock, stock_ticker) in enumerate(stocks):
        try:
            # fast_mode=True skips deep ML to make scanning 50 stocks much faster
            result = analyze_stock(stock, stock_ticker, market_name, market_score, fast_mode=True)
            if result:
                rows.append({
                    "Stock": stock,
                    "CMP": round(result["current"], 2),
                    "Signal": result["signal"],
                    "1D %": round(result["changes"]["1d"], 2),
                    "5D %": round(result["changes"]["5d"], 2),
                    "Stop Loss": round(result["levels"]["stop"], 2),
                    "R/R": round(result["levels"]["rr"], 2),
                    "Confidence": f'{result["confidence"]:.0f}%',
                })
        except Exception:
            pass
        progress.progress((i + 1) / len(stocks))
        
    progress.empty()
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No stock data available.")
    st.stop()


# ============================================================
# SINGLE STOCK ANALYSIS
# ============================================================

with st.spinner(f"Analyzing {symbol} with ML..."):
    result = analyze_stock(symbol, ticker, market_name, market_score, fast_mode=False)

if result is None:
    st.error("Unable to retrieve data for this stock.")
    st.stop()

# Extract Variables
data = result["data"]
last = result["last"]
current = result["current"]
today_change = result["today_change"]
signal = result["signal"]
predictions = result["predictions"]
changes = result["changes"]
levels = result["levels"]
live = result["live"]

# ============================================================
# STOCK HEADER
# ============================================================
price_class = "positive" if today_change >= 0 else "negative"
arrow = "↑" if today_change >= 0 else "↓"
market_status_str, status_detail = market_status()

quote_time = live.get("quote_time") if live else None
quote_text = quote_time.strftime("%d %b %Y, %I:%M:%S %p IST") if quote_time else "Unavailable"

st.markdown(f"""
    <div class="stock-box">
        <div class="stock-name">{html.escape(symbol)}</div>
        <div class="stock-symbol">NSE • {html.escape(ticker)}</div>
        <div style="color:#94a3b8; font-size:12px; margin-top:18px;">LAST TRADED PRICE</div>
        <div class="price">{money(current)}</div>
        <div class="{price_class}">{arrow} {abs(today_change):.2f}% today</div>
        <div style="margin-top:12px; font-size:13px;">
            <b>{"🟢" if market_status_str == "Market Open" else "⚪"} {market_status_str}</b>
            <span style="color:#64748b; margin-left:10px;">{status_detail}</span>
        </div>
        <div style="color:#64748b; font-size:11px; margin-top:8px;">Quote time: {quote_text}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# CSV EXPORT
# ============================================================
csv_data = data.to_csv().encode('utf-8')
st.download_button(
    label="💾 Download Historical & Technical Data (CSV)",
    data=csv_data,
    file_name=f'{symbol}_technical_data.csv',
    mime='text/csv',
)

# ============================================================
# SUMMARY
# ============================================================
st.subheader("📌 Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Signal", signal)
c2.metric("Confidence", f'{result["confidence"]:.0f}%')
c3.metric("Technical Score", result["tech"])
c4.metric("Market Trend", market_name)

# ============================================================
# AI PREDICTIONS
# ============================================================
st.subheader("🤖 AI Price Predictions")
p1, p2, p3 = st.columns(3)
p1.metric("1 Trading Day", money(predictions["1d"]), f'{changes["1d"]:+.2f}%')
p2.metric("5 Trading Days", money(predictions["5d"]), f'{changes["5d"]:+.2f}%')
p3.metric("20 Trading Days", money(predictions["20d"]), f'{changes["20d"]:+.2f}%')

# ============================================================
# TRADE PLAN & REASONING
# ============================================================
st.subheader("🎯 Trade Plan")
t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("Entry / CMP", money(current))
t2.metric("Target 1", money(levels["target1"]))
t3.metric("Target 2", money(levels["target2"]))
t4.metric("Stop Loss", money(levels["stop"]))
t5.metric("Risk / Reward", f'1 : {levels["rr"]:.2f}')

st.subheader("🧐 Why These Targets?")
atr = levels["atr_used"]
is_buy = signal in ["BUY", "STRONG BUY"]

target1_reason = f"**Target 1 ({money(levels['target1'])}):** Based on max of AI 1-day prediction ({money(predictions['1d'])}) and baseline momentum (CMP + 0.75x ATR = {money(current + (atr * 0.75))})." if is_buy else f"**Target 1 ({money(levels['target1'])}):** Based on min of AI 1-day prediction ({money(predictions['1d'])}) and baseline momentum (CMP - 0.75x ATR = {money(current - (atr * 0.75))})."
target2_reason = f"**Target 2 ({money(levels['target2'])}):** Pushes for a wider swing, taking the max of 1.5x ATR ({money(current + (atr * 1.5))}) or AI prediction with a 1.5% buffer." if is_buy else f"**Target 2 ({money(levels['target2'])}):** Pushes for a wider swing, taking the min of 1.5x ATR ({money(current - (atr * 1.5))}) or AI prediction with a 1.5% buffer."
stop_reason = f"**Stop Loss ({money(levels['stop'])}):** Placed at 1.2x ATR {'below' if is_buy else 'above'} current price to protect against normal market volatility ({money(atr)})."

st.markdown(f"""
<div class="reason">
    • {target1_reason}<br>
    • {target2_reason}<br>
    • {stop_reason}
</div>
""", unsafe_allow_html=True)

# ============================================================
# TECHNICAL INDICATORS
# ============================================================
st.subheader("📊 Technical Indicators")
a, b, c, d, e = st.columns(5)
a.metric("RSI", f'{safe_float(last["RSI"], 50):.1f}')
b.metric("SMA 20", money(last["SMA20"]))
c.metric("SMA 50", money(last["SMA50"]))
d.metric("SMA 200", money(last["SMA200"]))
e.metric("MACD", f'{safe_float(last["MACD"]):.2f}')

f, g, h, i, j = st.columns(5)
f.metric("ATR (Volatility)", money(last["ATR"]))
g.metric("Stochastic %K", f'{safe_float(last["STOCH_K"]):.1f}')
h.metric("BB Upper", money(last["BB_UPPER"]))
i.metric("BB Lower", money(last["BB_LOWER"]))
j.metric("Volume (20D Avg)", f'{last["VOL20"]:,.0f}')

# ============================================================
# CHART
# ============================================================
st.subheader("📈 Price Trend")
chart = data[["Close", "SMA20", "SMA50", "BB_UPPER", "BB_LOWER"]].tail(180)
st.line_chart(chart)

# ============================================================
# MODEL REASONING
# ============================================================
st.subheader("🧠 Why is the model predicting this?")
rsi = safe_float(last["RSI"], 50)
reasons = []

reasons.append("Price is above the 20-day moving average." if current > safe_float(last["SMA20"]) else "Price is below the 20-day moving average.")
reasons.append("Price is above the 50-day moving average." if current > safe_float(last["SMA50"]) else "Price is below the 50-day moving average.")
reasons.append("Medium-term trend is bullish." if safe_float(last["SMA50"]) > safe_float(last["SMA200"]) else "Medium-term trend is weak.")
reasons.append("MACD is bullish." if safe_float(last["MACD"]) > safe_float(last["MACD_SIGNAL"]) else "MACD is bearish.")

if rsi < 30:
    reasons.append(f"RSI is {rsi:.1f}, indicating oversold conditions.")
elif rsi > 70:
    reasons.append(f"RSI is {rsi:.1f}, indicating overbought conditions.")
else:
    reasons.append(f"RSI is {rsi:.1f}.")

reasons.append(f"Nifty overall market trend is currently {market_name}.")

for reason in reasons:
    st.markdown(f'<div class="reason">• {reason}</div>', unsafe_allow_html=True)

# ============================================================
# NEWS & SENTIMENT
# ============================================================
st.subheader("📰 Latest News & Sentiment")
n1, n2 = st.columns(2)
n1.metric("Overall Sentiment", result["sentiment"])
n2.metric("News Score", result["news_score"])

news = result["news"]
if news:
    for article in news[:6]:
        title = html.escape(article["title"])
        link = article["link"]
        date = html.escape(article["date"])
        st.markdown(f"""
            <div class="news">
                <b>{title}</b>
                <div style="color:#64748b; font-size:11px; margin-top:5px;">{date}</div>
                <a href="{link}" target="_blank" style="color:#60a5fa; font-size:12px;">Read article →</a>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("No recent news available.")

st.markdown("---")
st.caption("⚠️ This tool is for informational and educational purposes only. AI predictions are estimates and are not guaranteed. This is not financial advice.")
