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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Indian Stock AI Analyzer",
    page_icon="📈",
    layout="wide",
)


# =========================================================
# CONSTANTS
# =========================================================

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

NIFTY_10 = {
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "INFY": "INFY.NS",
    "TCS": "TCS.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "SBIN": "SBIN.NS",
    "AXISBANK": "AXISBANK.NS",
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
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "WIPRO": "WIPRO.NS",
}


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8fafc;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .stock-name {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
    }

    .stock-symbol {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 15px;
    }

    .cmp-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .cmp-price {
        font-size: 36px;
        font-weight: 800;
        color: #0f172a;
    }

    .positive {
        color: #16a34a;
        font-weight: 700;
    }

    .negative {
        color: #dc2626;
        font-weight: 700;
    }

    .neutral {
        color: #64748b;
        font-weight: 700;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .status-open {
        background: #dcfce7;
        color: #166534;
    }

    .status-closed {
        background: #f1f5f9;
        color: #475569;
    }

    .prediction-card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .prediction-title {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
    }

    .prediction-price {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 5px;
    }

    .prediction-change {
        font-size: 14px;
        font-weight: 700;
        margin-top: 5px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .reason-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value

    except Exception:
        return default


def money(value):
    value = safe_float(value)

    if value == 0:
        return "₹0.00"

    return f"₹{value:,.2f}"


def india_now():
    return datetime.now(IST)


def business_day_date():
    now = india_now()

    if now.weekday() >= 5:
        return now.date()

    return now.date()


# =========================================================
# FIXED MARKET STATUS
# =========================================================

def market_status_from_time(timestamp=None):
    """
    Determine NSE market status using India Standard Time.

    NSE regular trading session:
        Monday-Friday
        09:15 AM - 03:30 PM IST
    """

    now = india_now() if timestamp is None else timestamp

    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)

    now = now.astimezone(IST)

    # Weekend
    if now.weekday() >= 5:
        return "Market Closed", "Weekend"

    # Before market opens
    if now.time() < MARKET_OPEN:
        return "Market Closed", "Pre-market"

    # Regular market session
    if now.time() < MARKET_CLOSE:
        return "Market Open", "Live market"

    # After market closes
    return "Market Closed", "After market"


def signal_class(signal):
    if signal in ["BUY", "STRONG BUY"]:
        return "positive"

    if signal in ["SELL", "STRONG SELL"]:
        return "negative"

    return "neutral"


# =========================================================
# LIVE QUOTE
# =========================================================

@st.cache_data(ttl=30)
def fetch_live_quote(ticker):

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{ticker}?interval=1m&range=1d"
    )

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        response.raise_for_status()

        data = response.json()

        result = data["chart"]["result"][0]

        meta = result.get("meta", {})

        price = safe_float(
            meta.get("regularMarketPrice")
        )

        previous_close = safe_float(
            meta.get("previousClose")
        )

        regular_market_time = meta.get(
            "regularMarketTime"
        )

        # Fallback to latest 1-minute candle
        if price == 0:

            timestamps = result.get(
                "timestamp",
                []
            )

            indicators = (
                result
                .get("indicators", {})
                .get("quote", [])
            )

            if timestamps and indicators:

                closes = indicators[0].get(
                    "close",
                    []
                )

                for close in reversed(closes):

                    if close is not None:

                        price = safe_float(
                            close
                        )

                        break

        if price == 0:
            raise ValueError(
                "Unable to retrieve live price."
            )

        if previous_close == 0:
            previous_close = price

        change_pct = (
            ((price - previous_close)
             / previous_close) * 100
            if previous_close
            else 0
        )

        quote_time = None

        if regular_market_time:

            quote_time = datetime.fromtimestamp(
                regular_market_time,
                tz=IST
            )

        return {
            "price": price,
            "previous_close": previous_close,
            "change_pct": change_pct,
            "quote_time": quote_time,
            "source": "Yahoo Finance",
        }

    except Exception as e:

        return {
            "price": 0,
            "previous_close": 0,
            "change_pct": 0,
            "quote_time": None,
            "source": f"Error: {e}",
        }


# =========================================================
# HISTORICAL DATA
# =========================================================

@st.cache_data(ttl=900)
def download_data(ticker):

    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame()

    # Handle yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for col in required:

        if col not in df.columns:
            return pd.DataFrame()

    df = df[required].dropna()

    return df


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df):

    df = df.copy()

    # Moving averages
    df["SMA20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["SMA50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    df["SMA200"] = (
        df["Close"]
        .rolling(200)
        .mean()
    )

    # MACD
    ema12 = (
        df["Close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        df["Close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = (
        100 - (100 / (1 + rs))
    )

    # ATR
    high_low = (
        df["High"] - df["Low"]
    )

    high_close = (
        df["High"]
        - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"]
        - df["Close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = (
        true_range
        .rolling(14)
        .mean()
    )

    # Volume
    df["VOL20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    # Returns
    df["RETURN5"] = (
        df["Close"]
        .pct_change(5)
        * 100
    )

    df["RETURN20"] = (
        df["Close"]
        .pct_change(20)
        * 100
    )

    return df


# =========================================================
# NEWS
# =========================================================

def get_news(symbol, company_name):

    try:

        query = quote(
            f"{company_name} stock India"
        )

        url = (
            "https://news.google.com/rss/search?"
            f"q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        )

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        response.raise_for_status()

        root = ET.fromstring(
            response.content
        )

        news = []

        for item in root.findall(".//item")[:10]:

            title = item.findtext(
                "title",
                ""
            )

            link = item.findtext(
                "link",
                ""
            )

            pub_date = item.findtext(
                "pubDate",
                ""
            )

            title = html.unescape(
                title
            )

            news.append(
                {
                    "title": title,
                    "link": link,
                    "date": pub_date,
                }
            )

        return news

    except Exception:
        return []


def news_sentiment(news):

    if not news:
        return 0

    positive_words = [
        "gain",
        "growth",
        "profit",
        "surge",
        "rise",
        "strong",
        "bullish",
        "positive",
        "record",
        "upgrade",
    ]

    negative_words = [
        "fall",
        "drop",
        "loss",
        "weak",
        "decline",
        "bearish",
        "negative",
        "downgrade",
        "risk",
        "crisis",
    ]

    score = 0

    for item in news:

        title = item["title"].lower()

        for word in positive_words:

            if word in title:
                score += 1

        for word in negative_words:

            if word in title:
                score -= 1

    return score


# =========================================================
# TECHNICAL SCORE
# =========================================================

def technical_score(row):

    score = 0

    close = safe_float(row["Close"])

    sma20 = safe_float(row["SMA20"])
    sma50 = safe_float(row["SMA50"])
    sma200 = safe_float(row["SMA200"])

    rsi = safe_float(row["RSI"])

    macd = safe_float(row["MACD"])
    macd_signal = safe_float(
        row["MACD_SIGNAL"]
    )

    # Price vs moving averages
    if close > sma20:
        score += 1
    else:
        score -= 1

    if close > sma50:
        score += 1
    else:
        score -= 1

    if close > sma200:
        score += 1
    else:
        score -= 1

    # MACD
    if macd > macd_signal:
        score += 2
    else:
        score -= 2

    # RSI
    if 50 <= rsi <= 70:
        score += 2

    elif rsi > 70:
        score -= 1

    elif rsi < 30:
        score += 1

    else:
        score -= 1

    return score


# =========================================================
# MACHINE LEARNING
# =========================================================

FEATURES = [
    "Close",
    "SMA20",
    "SMA50",
    "SMA200",
    "MACD",
    "MACD_SIGNAL",
    "RSI",
    "ATR",
    "VOL20",
    "RETURN5",
    "RETURN20",
]


@st.cache_data(ttl=1800)
def ml_predictions(df):

    data = df.copy()

    data["TARGET_1D"] = (
        data["Close"]
        .shift(-1)
    )

    data["TARGET_5D"] = (
        data["Close"]
        .shift(-5)
    )

    data["TARGET_20D"] = (
        data["Close"]
        .shift(-20)
    )

    data = data.dropna()

    if len(data) < 100:
        return None

    X = data[FEATURES]

    latest = (
        df[FEATURES]
        .iloc[-1:]
        .copy()
    )

    predictions = {}

    for horizon, target in [
        ("1D", "TARGET_1D"),
        ("5D", "TARGET_5D"),
        ("20D", "TARGET_20D"),
    ]:

        y = data[target]

        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X,
            y,
        )

        prediction = model.predict(
            latest
        )[0]

        predictions[horizon] = float(
            prediction
        )

    return predictions


# =========================================================
# MARKET TREND
# =========================================================

@st.cache_data(ttl=900)
def market_trend():

    try:

        nifty = yf.download(
            "^NSEI",
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if nifty.empty:
            return "Neutral"

        if isinstance(
            nifty.columns,
            pd.MultiIndex
        ):
            nifty.columns = (
                nifty.columns
                .get_level_values(0)
            )

        close = nifty["Close"]

        sma20 = (
            close
            .rolling(20)
            .mean()
            .iloc[-1]
        )

        sma50 = (
            close
            .rolling(50)
            .mean()
            .iloc[-1]
        )

        current = close.iloc[-1]

        if current > sma20 and sma20 > sma50:
            return "Bullish"

        if current < sma20 and sma20 < sma50:
            return "Bearish"

        return "Neutral"

    except Exception:
        return "Neutral"


# =========================================================
# SIGNAL
# =========================================================

def generate_signal(
    technical,
    news_score,
    market,
    prediction_change,
):

    score = 0

    score += technical

    if news_score > 0:
        score += 1

    elif news_score < 0:
        score -= 1

    if market == "Bullish":
        score += 1

    elif market == "Bearish":
        score -= 1

    if prediction_change > 2:
        score += 2

    elif prediction_change > 0.5:
        score += 1

    elif prediction_change < -2:
        score -= 2

    elif prediction_change < -0.5:
        score -= 1

    if score >= 6:
        return "STRONG BUY"

    if score >= 3:
        return "BUY"

    if score <= -6:
        return "STRONG SELL"

    if score <= -3:
        return "SELL"

    return "HOLD"


# =========================================================
# TRADE LEVELS
# =========================================================

def trade_levels(
    cmp,
    predicted_price,
    atr,
    horizon="1D",
):

    cmp = safe_float(cmp)
    predicted_price = safe_float(
        predicted_price
    )
    atr = safe_float(atr)

    if cmp <= 0:
        return {
            "target1": 0,
            "target2": 0,
            "stop": 0,
            "risk": 0,
            "reward": 0,
            "rr": 0,
        }

    if atr <= 0:
        atr = cmp * 0.02

    multipliers = {
        "1D": 1.0,
        "5D": 1.5,
        "20D": 2.0,
    }

    multiplier = multipliers.get(
        horizon,
        1.0,
    )

    direction = (
        1
        if predicted_price >= cmp
        else -1
    )

    target1 = (
        cmp
        + direction
        * atr
        * multiplier
    )

    target2 = (
        cmp
        + direction
        * atr
        * multiplier
        * 2
    )

    stop = (
        cmp
        - direction
        * atr
        * multiplier
    )

    risk = abs(
        cmp - stop
    )

    reward = abs(
        target2 - cmp
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
        "rr": rr,
    }


# =========================================================
# CONFIDENCE
# =========================================================

def confidence(
    technical,
    news_score,
    market,
    prediction_change,
):

    value = 50

    value += min(
        abs(technical) * 4,
        20,
    )

    value += min(
        abs(news_score) * 3,
        10,
    )

    if market != "Neutral":
        value += 8

    value += min(
        abs(prediction_change),
        12,
    )

    return int(
        max(
            45,
            min(
                value,
                95,
            ),
        )
    )


# =========================================================
# PREDICTION REASON
# =========================================================

def prediction_reason(
    horizon,
    row,
    prediction,
    cmp,
):

    reasons = []

    close = safe_float(
        row["Close"]
    )

    sma20 = safe_float(
        row["SMA20"]
    )

    sma50 = safe_float(
        row["SMA50"]
    )

    sma200 = safe_float(
        row["SMA200"]
    )

    rsi = safe_float(
        row["RSI"]
    )

    macd = safe_float(
        row["MACD"]
    )

    macd_signal = safe_float(
        row["MACD_SIGNAL"]
    )

    change = (
        (
            prediction - cmp
        )
        / cmp
        * 100
        if cmp
        else 0
    )

    if close > sma20:
        reasons.append(
            "Price is above the 20-day moving average."
        )
    else:
        reasons.append(
            "Price is below the 20-day moving average."
        )

    if close > sma50:
        reasons.append(
            "Price remains above the 50-day moving average."
        )
    else:
        reasons.append(
            "Price is below the 50-day moving average."
        )

    if close > sma200:
        reasons.append(
            "Long-term trend is positive."
        )
    else:
        reasons.append(
            "Long-term trend remains weak."
        )

    if macd > macd_signal:
        reasons.append(
            "MACD is above its signal line."
        )
    else:
        reasons.append(
            "MACD is below its signal line."
        )

    if rsi >= 70:
        reasons.append(
            "RSI indicates an overbought condition."
        )
    elif rsi <= 30:
        reasons.append(
            "RSI indicates an oversold condition."
        )
    elif rsi >= 50:
        reasons.append(
            "RSI has positive momentum."
        )
    else:
        reasons.append(
            "RSI indicates relatively weak momentum."
        )

    if change > 0:
        reasons.append(
            f"Model estimates approximately {change:.2f}% upside over {horizon}."
        )
    else:
        reasons.append(
            f"Model estimates approximately {abs(change):.2f}% downside over {horizon}."
        )

    return reasons


# =========================================================
# ANALYZE STOCK
# =========================================================

def analyze_stock(
    ticker,
    company_name,
):

    # -----------------------------------------------------
    # Live quote
    # -----------------------------------------------------

    live = fetch_live_quote(
        ticker
    )

    # IMPORTANT:
    # Market status is calculated separately from the
    # cached Yahoo quote. This prevents stale "Pre-market"
    # / "Market Closed" values.
    market_status, status_detail = (
        market_status_from_time()
    )

    live["market_status"] = (
        market_status
    )

    live["status_detail"] = (
        status_detail
    )

    # -----------------------------------------------------
    # Historical data
    # -----------------------------------------------------

    df = download_data(
        ticker
    )

    if df.empty:
        return None

    df = add_indicators(
        df
    )

    df = df.dropna()

    if len(df) < 100:
        return None

    latest = df.iloc[-1]

    model_date = df.index[-1]

    # -----------------------------------------------------
    # CMP
    # -----------------------------------------------------

    cmp = safe_float(
        live["price"]
    )

    if cmp <= 0:
        cmp = safe_float(
            latest["Close"]
        )

    # -----------------------------------------------------
    # ML
    # -----------------------------------------------------

    predictions = ml_predictions(
        df
    )

    if predictions is None:
        return None

    # -----------------------------------------------------
    # Market trend
    # -----------------------------------------------------

    market = market_trend()

    # -----------------------------------------------------
    # News
    # -----------------------------------------------------

    news = get_news(
        ticker,
        company_name,
    )

    news_score = news_sentiment(
        news
    )

    # -----------------------------------------------------
    # Technical score
    # -----------------------------------------------------

    tech_score = technical_score(
        latest
    )

    # -----------------------------------------------------
    # Predictions
    #
    # IMPORTANT:
    # The ML model predicts based on the latest completed
    # daily candle. We convert the predicted percentage
    # move and apply it to the current live CMP.
    # -----------------------------------------------------

    prediction_prices = {}

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        model_prediction = safe_float(
            predictions[horizon]
        )

        model_close = safe_float(
            latest["Close"]
        )

        model_change = (
            (
                model_prediction
                - model_close
            )
            / model_close
            if model_close
            else 0
        )

        prediction_prices[horizon] = (
            cmp * (1 + model_change)
        )

    # -----------------------------------------------------
    # Signal
    # -----------------------------------------------------

    one_day_change = (
        (
            prediction_prices["1D"]
            - cmp
        )
        / cmp
        * 100
        if cmp
        else 0
    )

    signal = generate_signal(
        tech_score,
        news_score,
        market,
        one_day_change,
    )

    conf = confidence(
        tech_score,
        news_score,
        market,
        one_day_change,
    )

    # -----------------------------------------------------
    # Trade plans
    # -----------------------------------------------------

    levels = {}

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        levels[horizon] = trade_levels(
            cmp,
            prediction_prices[horizon],
            latest["ATR"],
            horizon,
        )

    # -----------------------------------------------------
    # Reasons
    # -----------------------------------------------------

    reasons = {}

    for horizon in [
        "1D",
        "5D",
        "20D",
    ]:

        reasons[horizon] = prediction_reason(
            horizon,
            latest,
            prediction_prices[horizon],
            cmp,
        )

    return {
        "ticker": ticker,
        "company": company_name,
        "df": df,
        "latest": latest,
        "model_date": model_date,
        "live": live,
        "cmp": cmp,
        "predictions": prediction_prices,
        "levels": levels,
        "signal": signal,
        "confidence": conf,
        "technical_score": tech_score,
        "market": market,
        "news": news,
        "news_score": news_score,
        "reasons": reasons,
    }


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Stock Analyzer")

watchlist_type = st.sidebar.selectbox(
    "Select Watchlist",
    [
        "NIFTY 10",
        "NIFTY 50",
        "Custom Stock",
    ],
)

if watchlist_type == "NIFTY 10":

    selected_symbol = st.sidebar.selectbox(
        "Select Stock",
        list(NIFTY_10.keys()),
    )

    ticker = NIFTY_10[
        selected_symbol
    ]

    company_name = selected_symbol

elif watchlist_type == "NIFTY 50":

    selected_symbol = st.sidebar.selectbox(
        "Select Stock",
        list(NIFTY_50.keys()),
    )

    ticker = NIFTY_50[
        selected_symbol
    ]

    company_name = selected_symbol

else:

    custom_symbol = st.sidebar.text_input(
        "NSE Symbol",
        value="ITC",
    ).strip().upper()

    if not custom_symbol:
        st.stop()

    ticker = (
        custom_symbol
        if custom_symbol.endswith(".NS")
        else f"{custom_symbol}.NS"
    )

    company_name = custom_symbol.replace(
        ".NS",
        "",
    )


analyze_button = st.sidebar.button(
    "🔍 Analyze Stock",
    use_container_width=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📈 Indian Stock AI Analyzer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">AI-powered technical, market and prediction analysis for Indian equities</div>',
    unsafe_allow_html=True,
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button or "analysis" not in st.session_state:

    with st.spinner(
        f"Analyzing {company_name}..."
    ):

        result = analyze_stock(
            ticker,
            company_name,
        )

    if result is None:

        st.error(
            "Unable to analyze this stock. "
            "Please check the symbol and try again."
        )

        st.stop()

    st.session_state[
        "analysis"
    ] = result

else:

    result = st.session_state[
        "analysis"
    ]


# =========================================================
# DATA
# =========================================================

live = result["live"]

cmp = result["cmp"]

change_pct = safe_float(
    live["change_pct"]
)

signal = result["signal"]

confidence_value = (
    result["confidence"]
)

predictions = result[
    "predictions"
]

levels = result[
    "levels"
]

latest = result[
    "latest"
]

news = result[
    "news"
]

market = result[
    "market"
]

reasons = result[
    "reasons"
]


# =========================================================
# STOCK HEADER
# =========================================================

col1, col2 = st.columns(
    [2.5, 1.5]
)

with col1:

    st.markdown(
        f"""
        <div class="stock-name">
            {company_name}
        </div>

        <div class="stock-symbol">
            NSE • {ticker}
        </div>

        <div class="cmp-label">
            Last Traded Price
        </div>

        <div class="cmp-price">
            {money(cmp)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    change_class = (
        "positive"
        if change_pct > 0
        else "negative"
        if change_pct < 0
        else "neutral"
    )

    arrow = (
        "↑"
        if change_pct > 0
        else "↓"
        if change_pct < 0
        else "→"
    )

    st.markdown(
        f"""
        <div class="{change_class}">
            {arrow} {abs(change_pct):.2f}% today
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_class = (
        "status-open"
        if live["market_status"]
        == "Market Open"
        else "status-closed"
    )

    quote_time = live.get(
        "quote_time"
    )

    quote_text = (
        quote_time.strftime(
            "%d %b %Y, %I:%M:%S %p IST"
        )
        if quote_time
        else "Unavailable"
    )

    model_date = result[
        "model_date"
    ]

    model_date_text = (
        model_date.strftime(
            "%d %b %Y"
        )
        if hasattr(
            model_date,
            "strftime",
        )
        else str(model_date)
    )

    st.markdown(
        f"""
        <div style="margin-top:12px;">
            <span class="status-pill {status_class}">
                ● {live["market_status"]}
            </span>
        </div>

        <div style="
            margin-top:8px;
            color:#64748b;
            font-size:11px;
        ">
            {live["status_detail"]}
        </div>

        <div style="
            margin-top:6px;
            color:#64748b;
            font-size:11px;
        ">
            Quote time:
            {quote_text}
        </div>

        <div style="
            margin-top:4px;
            color:#64748b;
            font-size:11px;
        ">
            Prediction model data through:
            {model_date_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        f"""
        <div class="prediction-card">

            <div class="prediction-title">
                Overall Signal
            </div>

            <div style="
                font-size:28px;
                font-weight:800;
                margin-top:8px;
            ">
                {signal}
            </div>

            <div style="
                margin-top:10px;
                color:#64748b;
                font-size:13px;
            ">
                Confidence
            </div>

            <div style="
                font-size:22px;
                font-weight:800;
                margin-top:3px;
            ">
                {confidence_value}%
            </div>

            <div style="
                margin-top:10px;
                color:#64748b;
                font-size:13px;
            ">
                Market Trend
            </div>

            <div style="
                font-size:18px;
                font-weight:700;
                margin-top:3px;
            ">
                {market}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PREDICTIONS
# =========================================================

st.markdown(
    '<div class="section-title">🤖 AI Price Predictions</div>',
    unsafe_allow_html=True,
)

prediction_cols = st.columns(3)

for col, horizon in zip(
    prediction_cols,
    ["1D", "5D", "20D"],
):

    predicted = predictions[
        horizon
    ]

    change = (
        (
            predicted - cmp
        )
        / cmp
        * 100
        if cmp
        else 0
    )

    change_class = (
        "positive"
        if change > 0
        else "negative"
        if change < 0
        else "neutral"
    )

    arrow = (
        "↑"
        if change > 0
        else "↓"
        if change < 0
        else "→"
    )

    with col:

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-title">
                    {horizon} Prediction
                </div>

                <div class="prediction-price">
                    {money(predicted)}
                </div>

                <div class="prediction-change {change_class}">
                    {arrow} {abs(change):.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# TRADE PLAN
# =========================================================

st.markdown(
    '<div class="section-title">🎯 Trade Plan</div>',
    unsafe_allow_html=True,
)

trade_cols = st.columns(3)

for col, horizon in zip(
    trade_cols,
    ["1D", "5D", "20D"],
):

    plan = levels[
        horizon
    ]

    with col:

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-title">
                    {horizon} Trade Setup
                </div>

                <div style="
                    margin-top:12px;
                    font-size:14px;
                    line-height:1.8;
                ">

                    <b>Target 1:</b>
                    {money(plan["target1"])}
                    <br>

                    <b>Target 2:</b>
                    {money(plan["target2"])}
                    <br>

                    <b>Stop Loss:</b>
                    {money(plan["stop"])}
                    <br>

                    <b>Risk:</b>
                    {money(plan["risk"])}
                    <br>

                    <b>Reward:</b>
                    {money(plan["reward"])}
                    <br>

                    <b>Risk / Reward:</b>
                    {plan["rr"]:.2f}

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PREDICTION REASONS
# =========================================================

st.markdown(
    '<div class="section-title">🧠 Why the Model Predicts This</div>',
    unsafe_allow_html=True,
)

for horizon in [
    "1D",
    "5D",
    "20D",
]:

    st.markdown(
        f"### {horizon}"
    )

    for reason in reasons[
        horizon
    ]:

        st.markdown(
            f"""
            <div class="reason-box">
                • {reason}
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

st.markdown(
    '<div class="section-title">📊 Technical Indicators</div>',
    unsafe_allow_html=True,
)

tech1, tech2, tech3, tech4, tech5 = st.columns(5)

with tech1:

    st.metric(
        "RSI",
        f"{safe_float(latest['RSI']):.1f}",
    )

with tech2:

    st.metric(
        "MACD",
        f"{safe_float(latest['MACD']):.2f}",
    )

with tech3:

    st.metric(
        "SMA 20",
        money(latest["SMA20"]),
    )

with tech4:

    st.metric(
        "SMA 50",
        money(latest["SMA50"]),
    )

with tech5:

    st.metric(
        "SMA 200",
        money(latest["SMA200"]),
    )


# =========================================================
# PRICE CHART
# =========================================================

st.markdown(
    '<div class="section-title">📈 Price Trend</div>',
    unsafe_allow_html=True,
)

chart_df = result[
    "df"
].copy()

chart_data = chart_df[
    [
        "Close",
        "SMA20",
        "SMA50",
        "SMA200",
    ]
].tail(180)

st.line_chart(
    chart_data
)


# =========================================================
# NEWS
# =========================================================

st.markdown(
    '<div class="section-title">📰 Latest News</div>',
    unsafe_allow_html=True,
)

if news:

    for item in news:

        title = item[
            "title"
        ]

        link = item[
            "link"
        ]

        st.markdown(
            f"- [{title}]({link})"
        )

else:

    st.info(
        "No recent news found."
    )


# =========================================================
# FACTORS
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Key Factors</div>',
    unsafe_allow_html=True,
)

factor1, factor2, factor3 = st.columns(3)

with factor1:

    st.metric(
        "Technical Score",
        result[
            "technical_score"
        ],
    )

with factor2:

    st.metric(
        "News Sentiment",
        result[
            "news_score"
        ],
    )

with factor3:

    st.metric(
        "Market Trend",
        market,
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.caption(
    """
    ⚠️ Disclaimer: This application provides
    educational and informational analysis only.
    AI predictions are estimates and are not
    guaranteed. This is not financial advice.
    Always perform your own research and consider
    your risk tolerance before making investment
    decisions.
    """
)
