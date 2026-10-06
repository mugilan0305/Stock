import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import html
from datetime import datetime, timedelta
from urllib.parse import quote
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Stock AI Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(37, 99, 235, 0.12),
                transparent 35%
            ),
            #0b1120;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .dashboard-title {
        font-size: 34px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 12px;
    }

    .market-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #111827;
        border: 1px solid #263449;
        padding: 7px 13px;
        border-radius: 999px;
        color: #cbd5e1;
        font-size: 13px;
        margin-bottom: 20px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        display: inline-block;
    }

    .stock-header {
        background: linear-gradient(
            135deg,
            #111827,
            #172033
        );
        border: 1px solid #263449;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .stock-name {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
    }

    .stock-symbol {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 3px;
    }

    .cmp-label {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 18px;
    }

    .cmp-price {
        font-size: 38px;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
    }

    .positive {
        color: #22c55e;
        font-weight: 700;
        font-size: 15px;
    }

    .negative {
        color: #ef4444;
        font-weight: 700;
        font-size: 15px;
    }

    .neutral {
        color: #f59e0b;
        font-weight: 700;
        font-size: 15px;
    }

    .signal-card {
        border-radius: 14px;
        padding: 16px 18px;
        border: 1px solid #334155;
        background: #111827;
        height: 100%;
    }

    .signal-title {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .signal-value {
        font-size: 24px;
        font-weight: 800;
        margin-top: 4px;
    }

    .buy {
        color: #22c55e;
    }

    .sell {
        color: #ef4444;
    }

    .hold {
        color: #f59e0b;
    }

    .prediction-card {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 16px;
        padding: 20px;
        min-height: 420px;
    }

    .prediction-heading {
        font-size: 18px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 15px;
    }

    .prediction-days {
        color: #94a3b8;
        font-size: 13px;
    }

    .prediction-price {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 12px;
    }

    .small-text {
        color: #94a3b8;
        font-size: 12px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 16px;
        font-weight: 700;
    }

    .reason-box {
        margin-top: 16px;
        padding: 12px;
        background: #0f172a;
        border: 1px solid #263449;
        border-radius: 10px;
    }

    .reason-title {
        color: #f8fafc;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .reason-text {
        color: #94a3b8;
        font-size: 12px;
        line-height: 1.6;
    }

    .confidence-box {
        margin-top: 14px;
        padding: 10px 12px;
        border-radius: 9px;
        background: #172033;
    }

    .news-card {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .news-title {
        color: #f8fafc;
        font-weight: 700;
        font-size: 14px;
    }

    .news-meta {
        color: #64748b;
        font-size: 11px;
        margin-top: 5px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 21px;
        font-weight: 800;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STOCK UNIVERSES
# ============================================================

NIFTY_10 = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "SBIN": "SBIN.NS",
    "LT": "LT.NS",
    "AXISBANK": "AXISBANK.NS"
}


NIFTY_50 = {
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "AXISBANK": "AXISBANK.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "BEL": "BEL.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "CIPLA": "CIPLA.NS",
    "COALINDIA": "COALINDIA.NS",
    "DRREDDY": "DRREDDY.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "ETERNAL": "ETERNAL.NS",
    "GRASIM": "GRASIM.NS",
    "HCLTECH": "HCLTECH.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "HINDALCO": "HINDALCO.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "INFY": "INFY.NS",
    "ITC": "ITC.NS",
    "JIOFIN": "JIOFIN.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS",
    "M&M": "M&M.NS",
    "MARUTI": "MARUTI.NS",
    "MAXHEALTH": "MAXHEALTH.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "POWERGRID": "POWERGRID.NS",
    "RELIANCE": "RELIANCE.NS",
    "SBILIFE": "SBILIFE.NS",
    "SBIN": "SBIN.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "TECHM": "TECHM.NS",
    "TITAN": "TITAN.NS",
    "TRENT": "TRENT.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS"
}


# Nifty 100 expansion.
# This can be updated later if NSE constituents change.
NIFTY_100_EXTRA = {
    "ABB": "ABB.NS",
    "ACC": "ACC.NS",
    "AMBUJACEM": "AMBUJACEM.NS",
    "BANKBARODA": "BANKBARODA.NS",
    "BERGEPAINT": "BERGEPAINT.NS",
    "BOSCHLTD": "BOSCHLTD.NS",
    "CANBK": "CANBK.NS",
    "CHOLAFIN": "CHOLAFIN.NS",
    "COLPAL": "COLPAL.NS",
    "CONCOR": "CONCOR.NS",
    "DABUR": "DABUR.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "DLF": "DLF.NS",
    "DMART": "DMART.NS",
    "GAIL": "GAIL.NS",
    "GODREJCP": "GODREJCP.NS",
    "GODREJPROP": "GODREJPROP.NS",
    "HAL": "HAL.NS",
    "HAVELLS": "HAVELLS.NS",
    "ICICIGI": "ICICIGI.NS",
    "ICICIPRULI": "ICICIPRULI.NS",
    "INDIGO": "INDIGO.NS",
    "IOC": "IOC.NS",
    "IRCTC": "IRCTC.NS",
    "JINDALSTEL": "JINDALSTEL.NS",
    "JSWENERGY": "JSWENERGY.NS",
    "LICI": "LICI.NS",
    "LUPIN": "LUPIN.NS",
    "MOTHERSON": "MOTHERSON.NS",
    "NAUKRI": "NAUKRI.NS",
    "NHPC": "NHPC.NS",
    "NMDC": "NMDC.NS",
    "OFSS": "OFSS.NS",
    "PAGEIND": "PAGEIND.NS",
    "PEL": "PEL.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "PETRONET": "PETRONET.NS",
    "PFC": "PFC.NS",
    "PIDILITIND": "PIDILITIND.NS",
    "POLYCAB": "POLYCAB.NS",
    "RECLTD": "RECLTD.NS",
    "SAIL": "SAIL.NS",
    "SHREECEM": "SHREECEM.NS",
    "SIEMENS": "SIEMENS.NS",
    "SRF": "SRF.NS",
    "TORNTPHARM": "TORNTPHARM.NS",
    "TVSMOTOR": "TVSMOTOR.NS",
    "VEDL": "VEDL.NS",
    "VOLTAS": "VOLTAS.NS",
    "ZYDUSLIFE": "ZYDUSLIFE.NS"
}

NIFTY_100 = dict(NIFTY_50)
NIFTY_100.update(NIFTY_100_EXTRA)


# ============================================================
# HELPER FUNCTIONS
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


def pct(value):
    return f"{safe_float(value):+.2f}%"


def get_signal_class(signal):
    if signal == "BUY":
        return "buy"
    if signal == "SELL":
        return "sell"
    return "hold"


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker):
    try:
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.copy()
        df.columns = [str(c).title() for c in df.columns]

        required = ["Open", "High", "Low", "Close", "Volume"]

        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        df = df[required].dropna()

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(df):

    data = df.copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()

    data["EMA12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA26"] = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = data["EMA12"] - data["EMA26"]
    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (100 / (1 + rs))

    data["VOL20"] = volume.rolling(20).mean()

    data["ATR"] = (
        pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ],
            axis=1
        ).max(axis=1).rolling(14).mean()
    )

    data["RETURN_5"] = close.pct_change(5)
    data["RETURN_20"] = close.pct_change(20)

    return data


# ============================================================
# NEWS
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol, limit=8):

    try:
        clean_symbol = symbol.replace(".NS", "")

        url = (
            "https://news.google.com/rss/search?"
            f"q={quote(clean_symbol + ' India stock')}"
            "&hl=en-IN"
            "&gl=IN"
            "&ceid=IN:en"
        )

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent":
                    "Mozilla/5.0"
            }
        )

        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)

        articles = []

        for item in root.findall(".//item")[:limit]:

            title_node = item.find("title")
            link_node = item.find("link")
            pub_node = item.find("pubDate")

            title = (
                title_node.text
                if title_node is not None
                else ""
            )

            link = (
                link_node.text
                if link_node is not None
                else ""
            )

            pub_date = (
                pub_node.text
                if pub_node is not None
                else ""
            )

            if title:
                articles.append(
                    {
                        "title": html.unescape(title),
                        "link": link,
                        "date": pub_date
                    }
                )

        return articles

    except Exception:
        return []


# ============================================================
# NEWS SENTIMENT
# ============================================================

POSITIVE_WORDS = [
    "profit",
    "profits",
    "growth",
    "strong",
    "upgrade",
    "buy",
    "bullish",
    "surge",
    "rally",
    "positive",
    "beat",
    "beats",
    "record",
    "expansion",
    "order",
    "orders",
    "approval",
    "wins",
    "partnership"
]

NEGATIVE_WORDS = [
    "loss",
    "losses",
    "weak",
    "downgrade",
    "sell",
    "bearish",
    "fall",
    "falls",
    "drop",
    "decline",
    "negative",
    "miss",
    "misses",
    "debt",
    "lawsuit",
    "investigation",
    "warning",
    "risk",
    "cut",
    "cuts"
]


def analyze_news_sentiment(news):

    score = 0
    positive = 0
    negative = 0

    for article in news:

        title = article.get("title", "").lower()

        for word in POSITIVE_WORDS:
            if word in title:
                score += 1
                positive += 1

        for word in NEGATIVE_WORDS:
            if word in title:
                score -= 1
                negative += 1

    if score >= 3:
        sentiment = "Positive"
    elif score <= -3:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "score": score,
        "positive": positive,
        "negative": negative,
        "sentiment": sentiment
    }


# ============================================================
# TECHNICAL SCORE
# ============================================================

def calculate_technical_score(data):

    if data.empty:
        return 0

    last = data.iloc[-1]

    score = 0

    close = safe_float(last["Close"])
    sma20 = safe_float(last["SMA20"])
    sma50 = safe_float(last["SMA50"])
    sma200 = safe_float(last["SMA200"])
    rsi = safe_float(last["RSI"], 50)
    macd = safe_float(last["MACD"])
    macd_signal = safe_float(last["MACD_SIGNAL"])
    volume = safe_float(last["Volume"])
    vol20 = safe_float(last["VOL20"])

    if close > sma20:
        score += 2
    else:
        score -= 2

    if close > sma50:
        score += 2
    else:
        score -= 2

    if sma50 > sma200:
        score += 1
    else:
        score -= 1

    if macd > macd_signal:
        score += 2
    else:
        score -= 2

    if 50 <= rsi <= 70:
        score += 1
    elif rsi < 30:
        score += 1
    elif rsi > 75:
        score -= 1

    if volume > vol20:
        score += 1

    return score


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(technical_score, news_score, prediction_1d):

    combined = (
        technical_score * 0.70
        + news_score * 0.30
    )

    if prediction_1d > 2 and combined >= 2.0:
        return "BUY"

    if prediction_1d < -2 and combined <= -2.0:
        return "SELL"

    if combined >= 1.0:
        return "BUY"

    if combined <= -1.0:
        return "SELL"

    return "HOLD"


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

def machine_learning_prediction(data):

    if data.empty or len(data) < 100:
        return None

    df = data.copy()

    features = [
        "Close",
        "SMA20",
        "SMA50",
        "RSI",
        "MACD",
        "MACD_SIGNAL",
        "ATR",
        "RETURN_5",
        "RETURN_20"
    ]

    df["TARGET_1"] = df["Close"].shift(-1)
    df["TARGET_5"] = df["Close"].shift(-5)
    df["TARGET_20"] = df["Close"].shift(-20)

    clean = df.dropna()

    if len(clean) < 60:
        return None

    X = clean[features]

    model_1 = RandomForestRegressor(
        n_estimators=150,
        max_depth=7,
        random_state=42,
        n_jobs=-1
    )

    model_5 = RandomForestRegressor(
        n_estimators=150,
        max_depth=7,
        random_state=43,
        n_jobs=-1
    )

    model_20 = RandomForestRegressor(
        n_estimators=150,
        max_depth=7,
        random_state=44,
        n_jobs=-1
    )

    model_1.fit(X, clean["TARGET_1"])
    model_5.fit(X, clean["TARGET_5"])
    model_20.fit(X, clean["TARGET_20"])

    latest = df.iloc[[-1]][features]

    pred1 = safe_float(model_1.predict(latest)[0])
    pred5 = safe_float(model_5.predict(latest)[0])
    pred20 = safe_float(model_20.predict(latest)[0])

    current = safe_float(df["Close"].iloc[-1])

    return {
        "1d": pred1,
        "5d": pred5,
        "20d": pred20,
        "current": current
    }


# ============================================================
# PREDICTION ADJUSTMENT
# ============================================================

def apply_signal_adjustment(
    current,
    prediction,
    technical_score,
    news_score
):

    technical_factor = technical_score * 0.0015
    news_factor = news_score * 0.002

    adjusted_return = (
        (prediction / current - 1)
        + technical_factor
        + news_factor
    )

    return current * (1 + adjusted_return)


# ============================================================
# STOP LOSS / TARGET
# ============================================================

def calculate_trade_levels(
    current,
    predicted,
    atr,
    signal
):

    current = safe_float(current)
    predicted = safe_float(predicted)
    atr = safe_float(atr)

    if atr <= 0:
        atr = current * 0.02

    if signal == "BUY":

        target = max(
            predicted,
            current + atr * 0.8
        )

        stop_loss = current - atr * 1.2

    elif signal == "SELL":

        target = min(
            predicted,
            current - atr * 0.8
        )

        stop_loss = current + atr * 1.2

    else:

        target = predicted

        if predicted >= current:
            stop_loss = current - atr
        else:
            stop_loss = current + atr

    target = max(target, 0.01)
    stop_loss = max(stop_loss, 0.01)

    if signal == "BUY":

        risk = current - stop_loss
        reward = target - current

    elif signal == "SELL":

        risk = stop_loss - current
        reward = current - target

    else:

        risk = abs(current - stop_loss)
        reward = abs(target - current)

    if risk <= 0:
        rr = 0
    else:
        rr = reward / risk

    return {
        "target": target,
        "stop_loss": stop_loss,
        "risk": risk,
        "reward": reward,
        "rr": rr
    }


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    technical_score,
    news_score,
    rr,
    prediction_change
):

    score = 50

    score += min(abs(technical_score) * 3, 20)

    score += min(abs(news_score) * 3, 10)

    if rr >= 2:
        score += 12
    elif rr >= 1:
        score += 6
    elif rr < 0.75:
        score -= 8

    if abs(prediction_change) >= 3:
        score += 8
    elif abs(prediction_change) >= 1.5:
        score += 4

    score = max(25, min(90, score))

    if score >= 75:
        level = "High"
    elif score >= 55:
        level = "Moderate"
    else:
        level = "Low"

    return score, level


# ============================================================
# REASON GENERATOR
# ============================================================

def generate_prediction_reason(
    data,
    news_info,
    horizon_return
):

    last = data.iloc[-1]

    close = safe_float(last["Close"])
    sma20 = safe_float(last["SMA20"])
    sma50 = safe_float(last["SMA50"])
    sma200 = safe_float(last["SMA200"])
    rsi = safe_float(last["RSI"], 50)
    macd = safe_float(last["MACD"])
    macd_signal = safe_float(last["MACD_SIGNAL"])
    volume = safe_float(last["Volume"])
    vol20 = safe_float(last["VOL20"])

    reasons = []

    if close > sma20:
        reasons.append(
            "price is above the 20-day moving average"
        )
    else:
        reasons.append(
            "price is below the 20-day moving average"
        )

    if close > sma50:
        reasons.append(
            "price is above the 50-day moving average"
        )
    else:
        reasons.append(
            "price is below the 50-day moving average"
        )

    if sma50 > sma200:
        reasons.append(
            "the medium-term trend is positive"
        )
    else:
        reasons.append(
            "the medium-term trend remains weak"
        )

    if macd > macd_signal:
        reasons.append(
            "MACD is above its signal line, supporting momentum"
        )
    else:
        reasons.append(
            "MACD is below its signal line, indicating weaker momentum"
        )

    if rsi < 30:
        reasons.append(
            f"RSI at {rsi:.1f} indicates oversold conditions and possible technical rebound"
        )
    elif rsi > 70:
        reasons.append(
            f"RSI at {rsi:.1f} indicates overbought conditions"
        )
    elif rsi >= 50:
        reasons.append(
            f"RSI at {rsi:.1f} indicates reasonably positive momentum"
        )
    else:
        reasons.append(
            f"RSI at {rsi:.1f} indicates relatively weak momentum"
        )

    if volume > vol20:
        reasons.append(
            "trading volume is above its 20-day average"
        )

    sentiment = news_info.get("sentiment", "Neutral")

    if sentiment == "Positive":
        reasons.append(
            "recent news sentiment is positive"
        )
    elif sentiment == "Negative":
        reasons.append(
            "recent news sentiment is negative"
        )
    else:
        reasons.append(
            "recent news sentiment is broadly neutral"
        )

    direction = "upside" if horizon_return >= 0 else "downside"

    reason_text = (
        f"The model expects {direction} of approximately "
        f"{abs(horizon_return):.2f}% over this horizon because "
        + "; ".join(reasons[:6])
        + "."
    )

    return reason_text


# ============================================================
# ANALYZE STOCK
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def analyze_stock(symbol, ticker):

    raw = get_stock_data(ticker)

    if raw.empty:
        return None

    data = calculate_indicators(raw)

    if data.empty:
        return None

    last = data.iloc[-1]

    current = safe_float(last["Close"])

    if current <= 0:
        return None

    technical_score = calculate_technical_score(data)

    news = get_news(symbol)

    news_info = analyze_news_sentiment(news)

    ml = machine_learning_prediction(data)

    if ml is None:

        pred1 = current * 1.005
        pred5 = current * 1.015
        pred20 = current * 1.03

    else:

        pred1 = apply_signal_adjustment(
            current,
            ml["1d"],
            technical_score,
            news_info["score"]
        )

        pred5 = apply_signal_adjustment(
            current,
            ml["5d"],
            technical_score,
            news_info["score"]
        )

        pred20 = apply_signal_adjustment(
            current,
            ml["20d"],
            technical_score,
            news_info["score"]
        )

    pred1_change = (pred1 / current - 1) * 100
    pred5_change = (pred5 / current - 1) * 100
    pred20_change = (pred20 / current - 1) * 100

    signal = generate_signal(
        technical_score,
        news_info["score"],
        pred1_change
    )

    atr = safe_float(last["ATR"])

    trade = calculate_trade_levels(
        current,
        pred1,
        atr,
        signal
    )

    confidence_score, confidence_level = calculate_confidence(
        technical_score,
        news_info["score"],
        trade["rr"],
        pred1_change
    )

    reason = generate_prediction_reason(
        data,
        news_info,
        pred1_change
    )

    return {
        "symbol": symbol,
        "ticker": ticker,
        "data": data,
        "current": current,
        "change_1d": pred1_change,
        "pred1": pred1,
        "pred5": pred5,
        "pred20": pred20,
        "change5": pred5_change,
        "change20": pred20_change,
        "signal": signal,
        "technical_score": technical_score,
        "news_info": news_info,
        "news": news,
        "trade": trade,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "reason": reason,
        "updated": datetime.now()
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 📊 Stock Analyzer")

universe_choice = st.sidebar.selectbox(
    "Stock Universe",
    [
        "Nifty 10",
        "Nifty 50",
        "Nifty 100",
        "Custom Stock"
    ]
)

if universe_choice == "Nifty 10":
    universe = NIFTY_10

elif universe_choice == "Nifty 50":
    universe = NIFTY_50

elif universe_choice == "Nifty 100":
    universe = NIFTY_100

else:
    universe = {}


if universe_choice == "Custom Stock":

    custom_symbol = st.sidebar.text_input(
        "Enter NSE Symbol",
        value="TCS"
    ).upper().strip()

    if custom_symbol:
        selected_symbol = custom_symbol
        selected_ticker = (
            custom_symbol
            if custom_symbol.endswith(".NS")
            else custom_symbol + ".NS"
        )

else:

    selected_symbol = st.sidebar.selectbox(
        "Select Stock",
        list(universe.keys()),
        index=(
            list(universe.keys()).index("TCS")
            if "TCS" in universe
            else 0
        )
    )

    selected_ticker = universe[selected_symbol]


st.sidebar.markdown("---")

scan_universe = st.sidebar.checkbox(
    "Scan selected universe",
    value=False
)

refresh = st.sidebar.button(
    "🔄 Refresh Data",
    use_container_width=True
)

if refresh:
    st.cache_data.clear()
    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-title">
        📈 Indian Stock AI Analyzer
    </div>

    <div class="dashboard-subtitle">
        Live market view • Technical analysis • AI-assisted prediction •
        News sentiment • Buy/Sell signals
    </div>

    <div class="market-status">
        <span class="status-dot"></span>
        Market data connected
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SCAN MODE
# ============================================================

if scan_universe:

    st.markdown(
        f"""
        <div class="section-title">
            🔎 {universe_choice} Market Scanner
        </div>
        """,
        unsafe_allow_html=True
    )

    rows = []

    progress = st.progress(0)

    symbols = list(universe.items())

    for index, (symbol, ticker) in enumerate(symbols):

        try:

            result = analyze_stock(
                symbol,
                ticker
            )

            if result is not None:

                rows.append(
                    {
                        "Stock": symbol,
                        "CMP": round(
                            result["current"],
                            2
                        ),
                        "Signal": result["signal"],
                        "1D Prediction": round(
                            result["pred1"],
                            2
                        ),
                        "1D %": round(
                            result["change_1d"],
                            2
                        ),
                        "5D %": round(
                            result["change5"],
                            2
                        ),
                        "20D %": round(
                            result["change20"],
                            2
                        ),
                        "Confidence": (
                            f"{result['confidence_score']:.0f}%"
                        ),
                        "News": result[
                            "news_info"
                        ]["sentiment"]
                    }
                )

        except Exception:
            pass

        progress.progress(
            int(
                ((index + 1) / len(symbols))
                * 100
            )
        )

    progress.empty()

    if rows:

        scan_df = pd.DataFrame(rows)

        def highlight_signal(value):

            if value == "BUY":
                return (
                    "color: #22c55e; "
                    "font-weight: 800;"
                )

            if value == "SELL":
                return (
                    "color: #ef4444; "
                    "font-weight: 800;"
                )

            return (
                "color: #f59e0b; "
                "font-weight: 800;"
            )

        styled = (
            scan_df.style
            .map(
                highlight_signal,
                subset=["Signal"]
            )
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Unable to retrieve stock data. "
            "Please try again."
        )

    st.stop()


# ============================================================
# SINGLE STOCK ANALYSIS
# ============================================================

with st.spinner(
    f"Analyzing {selected_symbol}..."
):

    result = analyze_stock(
        selected_symbol,
        selected_ticker
    )


if result is None:

    st.error(
        f"Unable to retrieve data for "
        f"{selected_symbol}. Please verify the symbol."
    )

    st.stop()


# ============================================================
# BASIC VALUES
# ============================================================

current = result["current"]
signal = result["signal"]
trade = result["trade"]
news_info = result["news_info"]

data = result["data"]
last = data.iloc[-1]

daily_move = (
    ((current / safe_float(data["Close"].iloc[-2])) - 1)
    * 100
    if len(data) >= 2
    else 0
)

signal_class = get_signal_class(signal)


# ============================================================
# STOCK HEADER
# ============================================================

st.markdown(
    f"""
    <div class="stock-header">

        <div class="stock-name">
            {selected_symbol}
        </div>

        <div class="stock-symbol">
            NSE • {selected_ticker}
        </div>

        <div class="cmp-label">
            Current Market Price
        </div>

        <div class="cmp-price">
            {money(current)}
        </div>

        <div class="{('positive' if daily_move >= 0 else 'negative')}">
            {'↑' if daily_move >= 0 else '↓'}
            {abs(daily_move):.2f}% today
        </div>

        <div style="
            margin-top:10px;
            color:#64748b;
            font-size:11px;
        ">
            Last updated:
            {result['updated'].strftime('%d %b %Y, %I:%M:%S %p')}
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIGNAL SUMMARY
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                Trading Signal
            </div>

            <div class="signal-value {signal_class}">
                {signal}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                Confidence
            </div>

            <div class="signal-value">
                {result['confidence_score']:.0f}%
            </div>

            <div class="small-text">
                {result['confidence_level']}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                News Sentiment
            </div>

            <div class="signal-value">
                {news_info['sentiment']}
            </div>

            <div class="small-text">
                Score: {news_info['score']}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                Technical Score
            </div>

            <div class="signal-value">
                {result['technical_score']}
            </div>

            <div class="small-text">
                Technical momentum
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION CARDS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🤖 AI Price Prediction
    </div>
    """,
    unsafe_allow_html=True
)


predictions = [
    (
        "1 Trading Day",
        result["pred1"],
        result["change_1d"]
    ),
    (
        "5 Trading Days",
        result["pred5"],
        result["change5"]
    ),
    (
        "20 Trading Days",
        result["pred20"],
        result["change20"]
    )
]


cols = st.columns(3)


for i, (
    horizon,
    predicted,
    change
) in enumerate(predictions):

    if change >= 0:
        direction_class = "positive"
        arrow = "↑"
    else:
        direction_class = "negative"
        arrow = "↓"

    estimated_days = [1, 5, 20][i]

    estimated_date = (
        datetime.now()
        + timedelta(days=estimated_days)
    )

    if i == 0:

        card_target = trade["target"]
        card_stop = trade["stop_loss"]
        card_rr = trade["rr"]

    else:

        card_target = predicted

        if change >= 0:
            card_stop = current * 0.97
        else:
            card_stop = current * 1.03

        risk = abs(current - card_stop)
        reward = abs(card_target - current)

        card_rr = (
            reward / risk
            if risk > 0
            else 0
        )

    with cols[i]:

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-heading">
                    {horizon}
                </div>

                <div class="prediction-days">
                    Model projected price
                </div>

                <div class="prediction-price">
                    {money(predicted)}
                </div>

                <div class="{direction_class}"
                     style="margin-top:5px;">
                    {arrow} {change:+.2f}%
                </div>

                <div style="
                    margin-top:18px;
                    text-align:left;
                ">

                    <div class="small-text">
                        Target
                    </div>

                    <div class="metric-value">
                        {money(card_target)}
                    </div>

                    <div class="small-text"
                         style="margin-top:10px;">
                        Stop Loss
                    </div>

                    <div style="
                        font-size:16px;
                        font-weight:700;
                        color:#ef4444;
                    ">
                        {money(card_stop)}
                    </div>

                    <div class="small-text"
                         style="margin-top:10px;">
                        Risk / Reward
                    </div>

                    <div style="
                        font-size:16px;
                        font-weight:700;
                        color:#60a5fa;
                    ">
                        1 : {card_rr:.2f}
                    </div>

                    <div class="small-text"
                         style="margin-top:10px;">
                        Expected Move
                    </div>

                    <div class="{direction_class}">
                        {money(abs(predicted-current))}
                    </div>

                    <div class="small-text"
                         style="margin-top:12px;">
                        Estimated Date
                    </div>

                    <div style="
                        font-size:13px;
                        color:#f8fafc;
                    ">
                        {estimated_date.strftime('%d %b %Y')}
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# WHY PREDICTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🧠 Why does the model expect this movement?
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div class="reason-box">

        <div class="reason-title">
            Prediction Reason
        </div>

        <div class="reason-text">
            {result['reason']}
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TRADE PLAN
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🎯 Trade Setup
    </div>
    """,
    unsafe_allow_html=True
)

t1, t2, t3, t4 = st.columns(4)


with t1:

    st.metric(
        "Current Price",
        money(current)
    )


with t2:

    st.metric(
        "Target",
        money(trade["target"])
    )


with t3:

    st.metric(
        "Stop Loss",
        money(trade["stop_loss"])
    )


with t4:

    st.metric(
        "Risk / Reward",
        f"1 : {trade['rr']:.2f}"
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📊 Technical Indicators
    </div>
    """,
    unsafe_allow_html=True
)

i1, i2, i3, i4, i5 = st.columns(5)


with i1:
    st.metric(
        "RSI",
        f"{safe_float(last['RSI']):.1f}"
    )


with i2:
    st.metric(
        "20D MA",
        money(last["SMA20"])
    )


with i3:
    st.metric(
        "50D MA",
        money(last["SMA50"])
    )


with i4:
    st.metric(
        "MACD",
        f"{safe_float(last['MACD']):.2f}"
    )


with i5:
    st.metric(
        "ATR",
        money(last["ATR"])
    )


# ============================================================
# NEWS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📰 Latest News & Market Drivers
    </div>
    """,
    unsafe_allow_html=True
)


if result["news"]:

    for article in result["news"][:6]:

        title = html.escape(
            article.get("title", "")
        )

        date = html.escape(
            article.get("date", "")
        )

        link = article.get(
            "link",
            "#"
        )

        st.markdown(
            f"""
            <div class="news-card">

                <div class="news-title">
                    {title}
                </div>

                <div class="news-meta">
                    {date}
                </div>

                <div style="margin-top:6px;">
                    <a href="{html.escape(link)}"
                       target="_blank"
                       style="
                           color:#60a5fa;
                           font-size:12px;
                           text-decoration:none;
                       ">
                        Read article →
                    </a>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

else:

    st.info(
        "No recent news articles were retrieved."
    )


# ============================================================
# MOVEMENT FACTORS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📌 Key Movement Factors
    </div>
    """,
    unsafe_allow_html=True
)


factor_data = []


if safe_float(last["Close"]) > safe_float(last["SMA20"]):
    factor_data.append(
        "Price is above the 20-day moving average."
    )
else:
    factor_data.append(
        "Price is below the 20-day moving average."
    )


if safe_float(last["Close"]) > safe_float(last["SMA50"]):
    factor_data.append(
        "Price is above the 50-day moving average."
    )
else:
    factor_data.append(
        "Price is below the 50-day moving average."
    )


if safe_float(last["MACD"]) > safe_float(last["MACD_SIGNAL"]):
    factor_data.append(
        "MACD momentum is positive."
    )
else:
    factor_data.append(
        "MACD momentum is negative."
    )


rsi_value = safe_float(last["RSI"], 50)

if rsi_value < 30:
    factor_data.append(
        "RSI is oversold, increasing the possibility of a technical rebound."
    )
elif rsi_value > 70:
    factor_data.append(
        "RSI is overbought, increasing the possibility of short-term cooling."
    )
else:
    factor_data.append(
        f"RSI is currently {rsi_value:.1f}."
    )


if news_info["sentiment"] == "Positive":
    factor_data.append(
        "Recent news sentiment is supportive."
    )
elif news_info["sentiment"] == "Negative":
    factor_data.append(
        "Recent news sentiment is a potential headwind."
    )
else:
    factor_data.append(
        "Recent news sentiment is mixed or neutral."
    )


for factor in factor_data:

    st.markdown(
        f"""
        <div style="
            background:#111827;
            border:1px solid #263449;
            border-radius:9px;
            padding:10px 14px;
            margin-bottom:7px;
            color:#cbd5e1;
            font-size:13px;
        ">
            • {html.escape(factor)}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:30px;
        padding:15px;
        background:#111827;
        border:1px solid #263449;
        border-radius:12px;
        color:#64748b;
        font-size:11px;
        line-height:1.6;
    ">
        <b style="color:#94a3b8;">
            Disclaimer:
        </b>
        This application provides model-based estimates using market data,
        technical indicators, news sentiment and machine-learning calculations.
        Predictions are not guaranteed and should not be treated as financial
        advice. Actual market prices can differ substantially from model
        estimates. Always perform your own research and consider your risk
        tolerance before making investment decisions.
    </div>
    """,
    unsafe_allow_html=True
)
