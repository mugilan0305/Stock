```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import html
import os
import json
from datetime import datetime, timedelta
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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        radial-gradient(
            circle at top left,
            rgba(37, 99, 235, 0.10),
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

    .title {
        font-size: 36px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .stock-card {
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
        font-size: 29px;
        font-weight: 800;
        color: #f8fafc;
    }

    .stock-symbol {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 3px;
    }

    .cmp-label {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 18px;
    }

    .cmp-price {
        font-size: 40px;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
    }

    .positive {
        color: #22c55e;
        font-weight: 700;
    }

    .negative {
        color: #ef4444;
        font-weight: 700;
    }

    .neutral {
        color: #f59e0b;
        font-weight: 700;
    }

    .info-card {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 16px;
        min-height: 105px;
    }

    .info-label {
        color: #94a3b8;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .info-value {
        color: #f8fafc;
        font-size: 23px;
        font-weight: 800;
        margin-top: 4px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    .prediction-card {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 16px;
        padding: 20px;
        min-height: 440px;
    }

    .prediction-heading {
        color: #f8fafc;
        font-size: 18px;
        font-weight: 800;
    }

    .prediction-price {
        color: #f8fafc;
        font-size: 29px;
        font-weight: 800;
        margin-top: 12px;
    }

    .small-text {
        color: #94a3b8;
        font-size: 12px;
    }

    .reason-box {
        background: #0f172a;
        border: 1px solid #263449;
        border-radius: 12px;
        padding: 15px;
        margin-top: 15px;
    }

    .reason-title {
        color: #f8fafc;
        font-weight: 800;
        font-size: 14px;
        margin-bottom: 7px;
    }

    .reason-text {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.65;
    }

    .signal-buy {
        color: #22c55e;
    }

    .signal-sell {
        color: #ef4444;
    }

    .signal-hold {
        color: #f59e0b;
    }

    .news-card {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 11px;
        padding: 14px;
        margin-bottom: 9px;
    }

    .news-title {
        color: #f8fafc;
        font-size: 14px;
        font-weight: 700;
    }

    .news-meta {
        color: #64748b;
        font-size: 11px;
        margin-top: 5px;
    }

    .factor {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 9px;
        padding: 11px 14px;
        margin-bottom: 7px;
        color: #cbd5e1;
        font-size: 13px;
    }

    .disclaimer {
        margin-top: 30px;
        padding: 15px;
        background: #111827;
        border: 1px solid #263449;
        border-radius: 12px;
        color: #64748b;
        font-size: 11px;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STOCK LISTS
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


# ============================================================
# GENERIC HELPERS
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


def signal_class(signal):
    if "BUY" in signal:
        return "signal-buy"
    if "SELL" in signal:
        return "signal-sell"
    return "signal-hold"


def business_days_from_now(days):
    return (
        pd.Timestamp.today()
        + pd.offsets.BDay(days)
    ).strftime("%d %b %Y")


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def download_data(ticker):

    try:

        df = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [
            str(c).title()
            for c in df.columns
        ]

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            column in df.columns
            for column in required
        ):
            return pd.DataFrame()

        return df[required].dropna()

    except Exception:
        return pd.DataFrame()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def indicators(df):

    data = df.copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()

    data["EMA12"] = close.ewm(
        span=12,
        adjust=False
    ).mean()

    data["EMA26"] = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = (
        data["EMA12"]
        - data["EMA26"]
    )

    data["MACD_SIGNAL"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = (
        avg_gain
        / avg_loss.replace(0, np.nan)
    )

    data["RSI"] = (
        100
        - (100 / (1 + rs))
    )

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    data["ATR"] = (
        true_range
        .rolling(14)
        .mean()
    )

    data["VOL20"] = (
        volume
        .rolling(20)
        .mean()
    )

    data["RETURN5"] = close.pct_change(5)
    data["RETURN20"] = close.pct_change(20)

    return data


# ============================================================
# NEWS
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(symbol):

    try:

        query = quote(
            f"{symbol} India stock"
        )

        url = (
            "https://news.google.com/rss/search?"
            f"q={query}"
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

        root = ET.fromstring(
            response.content
        )

        articles = []

        for item in root.findall(
            ".//item"
        )[:10]:

            title_node = item.find(
                "title"
            )

            link_node = item.find(
                "link"
            )

            date_node = item.find(
                "pubDate"
            )

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

            date = (
                date_node.text
                if date_node is not None
                else ""
            )

            if title:
                articles.append(
                    {
                        "title":
                        html.unescape(title),
                        "link": link,
                        "date": date
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
    "growth",
    "strong",
    "upgrade",
    "buy",
    "bullish",
    "surge",
    "rally",
    "positive",
    "beat",
    "record",
    "expansion",
    "order",
    "orders",
    "approval",
    "wins",
    "partnership",
    "outperform"
]

NEGATIVE_WORDS = [
    "loss",
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
    "debt",
    "lawsuit",
    "investigation",
    "warning",
    "risk",
    "cut",
    "cuts",
    "underperform"
]


def news_sentiment(news):

    score = 0

    positive_count = 0
    negative_count = 0

    for article in news:

        title = (
            article["title"]
            .lower()
        )

        for word in POSITIVE_WORDS:

            if word in title:
                score += 1
                positive_count += 1

        for word in NEGATIVE_WORDS:

            if word in title:
                score -= 1
                negative_count += 1

    if score >= 3:
        sentiment = "Positive"

    elif score <= -3:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return {
        "score": score,
        "positive": positive_count,
        "negative": negative_count,
        "sentiment": sentiment
    }


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
    signal = safe_float(
        last["MACD_SIGNAL"]
    )

    score = 0

    if close > sma20:
        score += 2
    else:
        score -= 2

    if close > sma50:
        score += 2
    else:
        score -= 2

    if sma50 > sma200:
        score += 2
    else:
        score -= 2

    if macd > signal:
        score += 2
    else:
        score -= 2

    if 50 <= rsi <= 70:
        score += 1

    elif rsi < 30:
        score += 1

    elif rsi > 75:
        score -= 1

    return score


# ============================================================
# MACHINE LEARNING
# ============================================================

def predict_prices(data):

    if len(data) < 150:
        return None

    df = data.copy()

    features = [
        "Close",
        "SMA20",
        "SMA50",
        "SMA200",
        "RSI",
        "MACD",
        "MACD_SIGNAL",
        "ATR",
        "RETURN5",
        "RETURN20"
    ]

    df["TARGET1"] = (
        df["Close"].shift(-1)
    )

    df["TARGET5"] = (
        df["Close"].shift(-5)
    )

    df["TARGET20"] = (
        df["Close"].shift(-20)
    )

    clean = df.dropna()

    if len(clean) < 100:
        return None

    X = clean[features]

    model1 = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    model5 = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=43,
        n_jobs=-1
    )

    model20 = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=44,
        n_jobs=-1
    )

    model1.fit(
        X,
        clean["TARGET1"]
    )

    model5.fit(
        X,
        clean["TARGET5"]
    )

    model20.fit(
        X,
        clean["TARGET20"]
    )

    latest = df.iloc[[-1]][features]

    return {
        "1d": safe_float(
            model1.predict(latest)[0]
        ),
        "5d": safe_float(
            model5.predict(latest)[0]
        ),
        "20d": safe_float(
            model20.predict(latest)[0]
        )
    }


# ============================================================
# MARKET TREND
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def market_trend():

    nifty = download_data(
        "^NSEI"
    )

    if nifty.empty:
        return "Unknown", 0

    nifty = indicators(nifty)

    last = nifty.iloc[-1]

    close = safe_float(last["Close"])
    sma20 = safe_float(last["SMA20"])
    sma50 = safe_float(last["SMA50"])

    score = 0

    if close > sma20:
        score += 1
    else:
        score -= 1

    if close > sma50:
        score += 1
    else:
        score -= 1

    if score == 2:
        return "Bullish", score

    if score == -2:
        return "Bearish", score

    return "Neutral", score


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(
    tech_score,
    news_score,
    predicted_change,
    market_score
):

    combined = (
        tech_score * 0.55
        + news_score * 0.20
        + np.sign(predicted_change)
        * min(abs(predicted_change), 5)
        * 0.25
        + market_score * 0.5
    )

    if combined >= 5:
        return "STRONG BUY"

    if combined >= 2:
        return "BUY"

    if combined <= -5:
        return "STRONG SELL"

    if combined <= -2:
        return "SELL"

    return "HOLD"


# ============================================================
# TRADE LEVELS
# ============================================================

def trade_levels(
    current,
    prediction,
    atr,
    signal
):

    current = safe_float(current)
    prediction = safe_float(prediction)
    atr = safe_float(atr)

    if atr <= 0:
        atr = current * 0.02

    if signal in [
        "BUY",
        "STRONG BUY"
    ]:

        target1 = max(
            prediction,
            current + atr * 0.75
        )

        target2 = max(
            current + atr * 1.5,
            prediction * 1.015
        )

        stop = current - atr * 1.2

    elif signal in [
        "SELL",
        "STRONG SELL"
    ]:

        target1 = min(
            prediction,
            current - atr * 0.75
        )

        target2 = min(
            current - atr * 1.5,
            prediction * 0.985
        )

        stop = current + atr * 1.2

    else:

        target1 = prediction
        target2 = prediction

        stop = (
            current - atr
            if prediction >= current
            else current + atr
        )

    target1 = max(
        target1,
        0.01
    )

    target2 = max(
        target2,
        0.01
    )

    stop = max(
        stop,
        0.01
    )

    if signal in [
        "BUY",
        "STRONG BUY"
    ]:

        risk = current - stop
        reward = target1 - current

    elif signal in [
        "SELL",
        "STRONG SELL"
    ]:

        risk = stop - current
        reward = current - target1

    else:

        risk = abs(
            current - stop
        )

        reward = abs(
            target1 - current
        )

    rr = (
        reward / risk
        if risk > 0
        else 0
    )

    return {
        "target1": target1,
        "target2": target2,
        "stop": stop,
        "risk": risk,
        "reward": reward,
        "rr": rr
    }


# ============================================================
# CONFIDENCE
# ============================================================

def confidence(
    tech_score,
    news_score,
    rr,
    predicted_change,
    market_score
):

    score = 50

    score += min(
        abs(tech_score) * 4,
        20
    )

    score += min(
        abs(news_score) * 2,
        10
    )

    score += abs(market_score) * 4

    if rr >= 2:
        score += 12

    elif rr >= 1:
        score += 6

    elif rr < 0.75:
        score -= 8

    if abs(predicted_change) >= 3:
        score += 8

    elif abs(predicted_change) >= 1.5:
        score += 4

    score = max(
        25,
        min(score, 92)
    )

    if score >= 75:
        level = "High"

    elif score >= 55:
        level = "Moderate"

    else:
        level = "Low"

    return score, level


# ============================================================
# REASON
# ============================================================

def prediction_reason(
    data,
    news_info,
    market_name,
    predicted_change
):

    last = data.iloc[-1]

    close = safe_float(
        last["Close"]
    )

    sma20 = safe_float(
        last["SMA20"]
    )

    sma50 = safe_float(
        last["SMA50"]
    )

    sma200 = safe_float(
        last["SMA200"]
    )

    rsi = safe_float(
        last["RSI"],
        50
    )

    macd = safe_float(
        last["MACD"]
    )

    macd_signal = safe_float(
        last["MACD_SIGNAL"]
    )

    volume = safe_float(
        last["Volume"]
    )

    volume20 = safe_float(
        last["VOL20"]
    )

    factors = []

    if close > sma20:
        factors.append(
            "price is above the 20-day moving average"
        )
    else:
        factors.append(
            "price is below the 20-day moving average"
        )

    if close > sma50:
        factors.append(
            "price is above the 50-day moving average"
        )
    else:
        factors.append(
            "price is below the 50-day moving average"
        )

    if sma50 > sma200:
        factors.append(
            "the medium-term trend is bullish"
        )
    else:
        factors.append(
            "the medium-term trend remains weak"
        )

    if macd > macd_signal:
        factors.append(
            "MACD is above its signal line"
        )
    else:
        factors.append(
            "MACD is below its signal line"
        )

    if rsi < 30:
        factors.append(
            f"RSI at {rsi:.1f} indicates oversold conditions"
        )

    elif rsi > 70:
        factors.append(
            f"RSI at {rsi:.1f} indicates overbought conditions"
        )

    else:
        factors.append(
            f"RSI is {rsi:.1f}"
        )

    if volume > volume20:
        factors.append(
            "volume is above its 20-day average"
        )

    if news_info["sentiment"] == "Positive":

        factors.append(
            "recent news sentiment is positive"
        )

    elif news_info["sentiment"] == "Negative":

        factors.append(
            "recent news sentiment is negative"
        )

    else:

        factors.append(
            "recent news sentiment is neutral"
        )

    factors.append(
        f"the broader Nifty trend is {market_name.lower()}"
    )

    direction = (
        "upside"
        if predicted_change >= 0
        else "downside"
    )

    return (
        f"The model projects approximately "
        f"{abs(predicted_change):.2f}% {direction}. "
        + "; ".join(factors[:7])
        + "."
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

HISTORY_FILE = "prediction_history.json"


def load_history():

    if not os.path.exists(
        HISTORY_FILE
    ):
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return []


def save_prediction(
    symbol,
    current,
    prediction,
    prediction_date,
    signal
):

    history = load_history()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    for item in history:

        if (
            item.get("symbol") == symbol
            and item.get("date") == today
        ):
            return

    history.append(
        {
            "symbol": symbol,
            "date": today,
            "current": current,
            "prediction": prediction,
            "prediction_date": prediction_date,
            "signal": signal
        }
    )

    history = history[-1000:]

    try:

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                history,
                f,
                indent=2
            )

    except Exception:
        pass


def accuracy_report():

    history = load_history()

    if not history:
        return None

    checked = []

    for item in history:

        prediction_date = item.get(
            "prediction_date"
        )

        if not prediction_date:
            continue

        try:

            target_date = pd.Timestamp(
                prediction_date
            )

        except Exception:

            continue

        if target_date > pd.Timestamp.today():
            continue

        ticker = (
            item["symbol"] + ".NS"
        )

        data = download_data(
            ticker
        )

        if data.empty:
            continue

        actual = safe_float(
            data["Close"].iloc[-1]
        )

        predicted = safe_float(
            item["prediction"]
        )

        if predicted == 0:
            continue

        predicted_direction = (
            predicted
            > item["current"]
        )

        actual_direction = (
            actual
            > item["current"]
        )

        correct = (
            predicted_direction
            == actual_direction
        )

        checked.append(
            {
                "symbol":
                    item["symbol"],
                "date":
                    item["date"],
                "correct":
                    correct,
                "predicted":
                    predicted,
                "actual":
                    actual
            }
        )

    if not checked:
        return None

    accuracy = (
        sum(
            x["correct"]
            for x in checked
        )
        / len(checked)
        * 100
    )

    return {
        "accuracy": accuracy,
        "checked": len(checked)
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 📊 Stock Analyzer"
)

universe_name = st.sidebar.selectbox(
    "Stock Universe",
    [
        "Nifty 10",
        "Nifty 50",
        "Custom Stock"
    ]
)

if universe_name == "Nifty 10":

    universe = NIFTY_10

elif universe_name == "Nifty 50":

    universe = NIFTY_50

else:

    universe = {}


if universe_name == "Custom Stock":

    selected_symbol = (
        st.sidebar.text_input(
            "NSE Symbol",
            "TCS"
        )
        .upper()
        .strip()
    )

    selected_ticker = (
        selected_symbol
        if selected_symbol.endswith(
            ".NS"
        )
        else selected_symbol + ".NS"
    )

else:

    selected_symbol = (
        st.sidebar.selectbox(
            "Select Stock",
            list(universe.keys()),
            index=(
                list(
                    universe.keys()
                ).index("TCS")
                if "TCS" in universe
                else 0
            )
        )
    )

    selected_ticker = universe[
        selected_symbol
    ]


scan = st.sidebar.checkbox(
    "Scan entire universe"
)

if st.sidebar.button(
    "🔄 Refresh Data",
    use_container_width=True
):

    st.cache_data.clear()
    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="title">
        📈 Indian Stock AI Analyzer
    </div>

    <div class="subtitle">
        Technical analysis • Machine learning • News sentiment •
        Market trend • Risk management
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MARKET TREND
# ============================================================

market_name, market_score = (
    market_trend()
)


# ============================================================
# SCANNER
# ============================================================

if scan:

    st.markdown(
        f"""
        <div class="section-title">
            🔎 {universe_name} Scanner
        </div>
        """,
        unsafe_allow_html=True
    )

    rows = []

    progress = st.progress(0)

    symbols = list(
        universe.items()
    )

    for index, (
        symbol,
        ticker
    ) in enumerate(symbols):

        try:

            raw = download_data(
                ticker
            )

            if raw.empty:
                continue

            data = indicators(raw)

            last = data.iloc[-1]

            tech = technical_score(
                data
            )

            news = fetch_news(
                symbol
            )

            news_info = (
                news_sentiment(news)
            )

            ml = predict_prices(
                data
            )

            current = safe_float(
                last["Close"]
            )

            if ml:

                predicted = ml["1d"]

            else:

                predicted = current

            change = (
                (predicted / current)
                - 1
            ) * 100

            signal = generate_signal(
                tech,
                news_info["score"],
                change,
                market_score
            )

            conf, level = confidence(
                tech,
                news_info["score"],
                1,
                change,
                market_score
            )

            rows.append(
                {
                    "Stock": symbol,
                    "CMP": round(
                        current,
                        2
                    ),
                    "Signal": signal,
                    "1D Prediction":
                        round(
                            predicted,
                            2
                        ),
                    "Expected %":
                        round(
                            change,
                            2
                        ),
                    "Confidence":
                        f"{conf:.0f}%",
                    "News":
                        news_info[
                            "sentiment"
                        ]
                }
            )

        except Exception:
            pass

        progress.progress(
            int(
                (
                    (index + 1)
                    / len(symbols)
                ) * 100
            )
        )

    progress.empty()

    if rows:

        scan_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            scan_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No stock data could be retrieved."
        )

    st.stop()


# ============================================================
# ANALYZE SELECTED STOCK
# ============================================================

with st.spinner(
    f"Analyzing {selected_symbol}..."
):

    raw = download_data(
        selected_ticker
    )


if raw.empty:

    st.error(
        f"Unable to retrieve data for "
        f"{selected_symbol}."
    )

    st.stop()


data = indicators(raw)

last = data.iloc[-1]

current = safe_float(
    last["Close"]
)

previous = safe_float(
    data["Close"].iloc[-2]
)

today_change = (
    (current / previous)
    - 1
) * 100


tech_score = technical_score(
    data
)

news = fetch_news(
    selected_symbol
)

news_info = news_sentiment(
    news
)

ml = predict_prices(
    data
)

if ml:

    prediction1 = ml["1d"]
    prediction5 = ml["5d"]
    prediction20 = ml["20d"]

else:

    prediction1 = current * 1.005
    prediction5 = current * 1.015
    prediction20 = current * 1.03


prediction_change1 = (
    (prediction1 / current)
    - 1
) * 100

prediction_change5 = (
    (prediction5 / current)
    - 1
) * 100

prediction_change20 = (
    (prediction20 / current)
    - 1
) * 100


signal = generate_signal(
    tech_score,
    news_info["score"],
    prediction_change1,
    market_score
)


levels = trade_levels(
    current,
    prediction1,
    safe_float(last["ATR"]),
    signal
)


confidence_score, confidence_level = (
    confidence(
        tech_score,
        news_info["score"],
        levels["rr"],
        prediction_change1,
        market_score
    )
)


reason = prediction_reason(
    data,
    news_info,
    market_name,
    prediction_change1
)


last_updated = datetime.now().strftime(
    "%d %b %Y, %I:%M:%S %p"
)


# ============================================================
# SAVE PREDICTION
# ============================================================

save_prediction(
    selected_symbol,
    current,
    prediction1,
    (
        datetime.now()
        + timedelta(days=1)
    ).strftime("%Y-%m-%d"),
    signal
)


# ============================================================
# CURRENT MARKET PRICE
# ============================================================

price_class = (
    "positive"
    if today_change >= 0
    else "negative"
)

arrow = (
    "↑"
    if today_change >= 0
    else "↓"
)

st.markdown(
    f"""
    <div class="stock-card">

        <div class="stock-name">
            {html.escape(selected_symbol)}
        </div>

        <div class="stock-symbol">
            NSE • {html.escape(selected_ticker)}
        </div>

        <div class="cmp-label">
            Current Market Price
        </div>

        <div class="cmp-price">
            {money(current)}
        </div>

        <div class="{price_class}">
            {arrow} {abs(today_change):.2f}% today
        </div>

        <div style="
            margin-top:10px;
            color:#64748b;
            font-size:11px;
        ">
            Last updated:
            {last_updated}
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SUMMARY CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Trading Signal
            </div>

            <div class="info-value {signal_class(signal)}">
                {signal}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Confidence
            </div>

            <div class="info-value">
                {confidence_score:.0f}%
            </div>

            <div class="small-text">
                {confidence_level}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Technical Score
            </div>

            <div class="info-value">
                {tech_score}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Market Trend
            </div>

            <div class="info-value">
                {market_name}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# AI PREDICTIONS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🤖 AI Price Predictions
    </div>
    """,
    unsafe_allow_html=True
)


prediction_items = [
    (
        "1 Trading Day",
        prediction1,
        prediction_change1,
        1
    ),
    (
        "5 Trading Days",
        prediction5,
        prediction_change5,
        5
    ),
    (
        "20 Trading Days",
        prediction20,
        prediction_change20,
        20
    )
]


prediction_cols = st.columns(3)


for i, item in enumerate(
    prediction_items
):

    horizon = item[0]
    predicted = item[1]
    change = item[2]
    days = item[3]

    cls = (
        "positive"
        if change >= 0
        else "negative"
    )

    arrow = (
        "↑"
        if change >= 0
        else "↓"
    )

    if i == 0:

        target1 = levels[
            "target1"
        ]

        target2 = levels[
            "target2"
        ]

        stop = levels[
            "stop"
        ]

        rr = levels[
            "rr"
        ]

    else:

        target1 = predicted
        target2 = predicted

        if change >= 0:

            stop = current * 0.97

        else:

            stop = current * 1.03

        risk = abs(
            current - stop
        )

        reward = abs(
            predicted - current
        )

        rr = (
            reward / risk
            if risk > 0
            else 0
        )

    with prediction_cols[i]:

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-heading">
                    {horizon}
                </div>

                <div class="small-text">
                    Model projected price
                </div>

                <div class="prediction-price">
                    {money(predicted)}
                </div>

                <div class="{cls}"
                     style="margin-top:5px;">
                    {arrow}
                    {abs(change):.2f}%
                </div>

                <div style="
                    margin-top:18px;
                ">

                    <div class="small-text">
                        Target 1
                    </div>

                    <div style="
                        color:#22c55e;
                        font-size:16px;
                        font-weight:700;
                    ">
                        {money(target1)}
                    </div>

                    <div class="small-text"
                         style="margin-top:10px;">
                        Target 2
                    </div>

                    <div style="
                        color:#22c55e;
                        font-size:16px;
                        font-weight:700;
                    ">
                        {money(target2)}
                    </div>

                    <div class="small-text"
                         style="margin-top:10px;">
                        Stop Loss
                    </div>

                    <div style="
                        color:#ef4444;
                        font-size:16px;
                        font-weight:700;
                    ">
                        {money(stop)}
                    </div>

                    <div class="small-text"
                         style="margin-top:10px;">
                        Risk / Reward
                    </div>

                    <div style="
                        color:#60a5fa;
                        font-size:16px;
                        font-weight:700;
                    ">
                        1 : {rr:.2f}
                    </div>

                    <div class="small-text"
                         style="margin-top:10px;">
                        Estimated Date
                    </div>

                    <div style="
                        color:#f8fafc;
                        font-size:13px;
                    ">
                        {business_days_from_now(days)}
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
            {html.escape(reason)}
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
        🎯 Trade Plan
    </div>
    """,
    unsafe_allow_html=True
)

t1, t2, t3, t4, t5 = st.columns(5)


with t1:
    st.metric(
        "Entry / CMP",
        money(current)
    )


with t2:
    st.metric(
        "Target 1",
        money(levels["target1"])
    )


with t3:
    st.metric(
        "Target 2",
        money(levels["target2"])
    )


with t4:
    st.metric(
        "Stop Loss",
        money(levels["stop"])
    )


with t5:
    st.metric(
        "Risk / Reward",
        f"1 : {levels['rr']:.2f}"
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

a, b, c, d, e, f = st.columns(6)


with a:
    st.metric(
        "RSI",
        f"{safe_float(last['RSI']):.1f}"
    )


with b:
    st.metric(
        "20D MA",
        money(last["SMA20"])
    )


with c:
    st.metric(
        "50D MA",
        money(last["SMA50"])
    )


with d:
    st.metric(
        "200D MA",
        money(last["SMA200"])
    )


with e:
    st.metric(
        "MACD",
        f"{safe_float(last['MACD']):.2f}"
    )


with f:
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
        📰 Latest News & Movement Drivers
    </div>
    """,
    unsafe_allow_html=True
)


if news:

    for article in news[:6]:

        title = html.escape(
            article["title"]
        )

        link = html.escape(
            article["link"]
        )

        date = html.escape(
            article["date"]
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
                    <a href="{link}"
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
        "No recent news was retrieved."
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


factors = []


if current > safe_float(
    last["SMA20"]
):

    factors.append(
        "Price is above the 20-day moving average."
    )

else:

    factors.append(
        "Price is below the 20-day moving average."
    )


if current > safe_float(
    last["SMA50"]
):

    factors.append(
        "Price is above the 50-day moving average."
    )

else:

    factors.append(
        "Price is below the 50-day moving average."
    )


if safe_float(
    last["MACD"]
) > safe_float(
    last["MACD_SIGNAL"]
):

    factors.append(
        "MACD is supporting bullish momentum."
    )

else:

    factors.append(
        "MACD is showing bearish momentum."
    )


rsi = safe_float(
    last["RSI"],
    50
)

if rsi < 30:

    factors.append(
        "RSI is oversold and a technical rebound is possible."
    )

elif rsi > 70:

    factors.append(
        "RSI is overbought and short-term cooling is possible."
    )

else:

    factors.append(
        f"RSI is currently {rsi:.1f}."
    )


if news_info["sentiment"] == "Positive":

    factors.append(
        "Recent news sentiment is positive."
    )

elif news_info["sentiment"] == "Negative":

    factors.append(
        "Recent news sentiment is negative."
    )

else:

    factors.append(
        "Recent news sentiment is neutral or mixed."
    )


factors.append(
    f"Nifty market trend is currently {market_name}."
)


for factor in factors:

    st.markdown(
        f"""
        <div class="factor">
            • {html.escape(factor)}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION ACCURACY
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🎯 Model Prediction Accuracy
    </div>
    """,
    unsafe_allow_html=True
)

accuracy = accuracy_report()

if accuracy:

    st.metric(
        "Historical Directional Accuracy",
        f"{accuracy['accuracy']:.1f}%"
    )

    st.caption(
        f"Based on {accuracy['checked']} "
        "completed predictions."
    )

else:

    st.info(
        "Prediction accuracy will appear after "
        "the application has accumulated completed predictions."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        <b style="color:#94a3b8;">
            Disclaimer:
        </b>

        This application uses historical market data,
        technical indicators, news sentiment and machine-learning
        models to generate estimates. Predictions are not guaranteed
        and should not be treated as financial advice. Actual market
        prices may differ materially from model estimates. Always
        perform your own research and consider your risk tolerance
        before making investment decisions.

    </div>
    """,
    unsafe_allow_html=True
)
```
