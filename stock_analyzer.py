import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import html
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Indian Stock AI Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
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

    .stock-card,
    .info-card,
    .prediction-card,
    .reason-box,
    .news-card,
    .factor,
    .disclaimer {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 16px;
    }

    .stock-card {
        padding: 24px;
        margin-bottom: 20px;
    }

    .stock-name {
        font-size: 30px;
        font-weight: 800;
    }

    .stock-symbol,
    .cmp-label,
    .small-text {
        color: #94a3b8;
        font-size: 12px;
    }

    .cmp-label {
        margin-top: 18px;
    }

    .cmp-price {
        font-size: 40px;
        font-weight: 800;
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

    .signal-buy {
        color: #22c55e;
    }

    .signal-sell {
        color: #ef4444;
    }

    .signal-hold {
        color: #f59e0b;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    .prediction-card {
        padding: 20px;
        min-height: 520px;
    }

    .prediction-heading {
        font-size: 18px;
        font-weight: 800;
    }

    .prediction-price {
        font-size: 29px;
        font-weight: 800;
        margin-top: 10px;
    }

    .reason-box {
        padding: 16px;
    }

    .reason-title {
        font-weight: 800;
        margin-bottom: 7px;
    }

    .reason-text {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.65;
    }

    .news-card {
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
        padding: 11px 14px;
        margin-bottom: 7px;
        color: #cbd5e1;
        font-size: 13px;
    }

    .disclaimer {
        margin-top: 30px;
        padding: 15px;
        color: #64748b;
        font-size: 11px;
        line-height: 1.6;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 10px;
    }

    .status-open {
        background: rgba(34,197,94,.12);
        color: #22c55e;
    }

    .status-closed {
        background: rgba(245,158,11,.12);
        color: #f59e0b;
    }

    .status-unavailable {
        background: rgba(239,68,68,.12);
        color: #ef4444;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
}

# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
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


def business_day_date(days):
    return (
        pd.Timestamp.today().normalize()
        + pd.offsets.BDay(days)
    ).strftime("%d %b %Y")


def india_now():
    return datetime.now(
        ZoneInfo("Asia/Kolkata")
    )


def market_status_from_time(timestamp=None):
    now = (
        india_now()
        if timestamp is None
        else timestamp
    )

    if now.tzinfo is None:
        now = now.replace(
            tzinfo=ZoneInfo("Asia/Kolkata")
        )

    now = now.astimezone(
        ZoneInfo("Asia/Kolkata")
    )

    if now.weekday() >= 5:
        return "Market Closed", "Weekend"

    if (
        now.hour < 9
        or (
            now.hour == 9
            and now.minute < 15
        )
    ):
        return "Market Closed", "Pre-market"

    if (
        now.hour > 15
        or (
            now.hour == 15
            and now.minute >= 30
        )
    ):
        return "Market Closed", "After market"

    return "Market Open", "Live market"


# ============================================================
# LIVE PRICE FEED
# KEPT SEPARATE FROM THE PREDICTION ENGINE
# ============================================================

@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def fetch_live_quote(ticker):
    empty = {
        "price": None,
        "previous_close": None,
        "change_pct": None,
        "quote_time": None,
        "market_status": "Unavailable",
        "status_detail": "Live quote unavailable",
        "source": "Yahoo Finance",
    }

    try:
        url = (
            "https://query1.finance.yahoo.com/"
            "v8/finance/chart/"
            f"{quote(ticker)}"
        )

        params = {
            "interval": "1m",
            "range": "1d",
            "includePrePost": "false",
            "events": "div,splits",
        }

        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        response.raise_for_status()

        payload = response.json()

        results = (
            payload
            .get("chart", {})
            .get("result")
            or []
        )

        if not results:
            return empty

        meta = results[0].get(
            "meta",
            {},
        )

        price = safe_float(
            meta.get(
                "regularMarketPrice"
            ),
            np.nan,
        )

        previous_close = safe_float(
            meta.get(
                "previousClose",
                meta.get(
                    "chartPreviousClose"
                ),
            ),
            np.nan,
        )

        regular_market_time = meta.get(
            "regularMarketTime"
        )

        quote_time = None

        if regular_market_time:
            quote_time = datetime.fromtimestamp(
                int(regular_market_time),
                tz=ZoneInfo("Asia/Kolkata"),
            )

        # Fallback to the latest 1-minute close.
        if not np.isfinite(price):
            timestamps = (
                results[0]
                .get("timestamp")
                or []
            )

            indicators = results[0].get(
                "indicators",
                {},
            )

            quote_rows = (
                indicators.get("quote")
                or []
            )

            closes = (
                quote_rows[0].get(
                    "close",
                    [],
                )
                if quote_rows
                else []
            )

            for ts, close_value in reversed(
                list(
                    zip(
                        timestamps,
                        closes,
                    )
                )
            ):
                candidate = safe_float(
                    close_value,
                    np.nan,
                )

                if np.isfinite(candidate):
                    price = candidate

                    quote_time = (
                        datetime.fromtimestamp(
                            int(ts),
                            tz=ZoneInfo(
                                "Asia/Kolkata"
                            ),
                        )
                    )

                    break

        if not np.isfinite(price):
            return empty

        if (
            not np.isfinite(
                previous_close
            )
            or previous_close <= 0
        ):
            previous_close = None

        change_pct = None

        if previous_close:
            change_pct = (
                (
                    price
                    / previous_close
                )
                - 1.0
            ) * 100.0

        status, detail = (
            market_status_from_time()
        )

        return {
            "price": float(price),
            "previous_close": (
                previous_close
            ),
            "change_pct": change_pct,
            "quote_time": quote_time,
            "market_status": status,
            "status_detail": detail,
            "source": "Yahoo Finance",
        }

    except Exception:
        return empty


# ============================================================
# HISTORICAL DATA
# USED ONLY BY THE PREDICTION ENGINE
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def download_data(ticker):
    try:
        df = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(
            df.columns,
            pd.MultiIndex,
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        df.columns = [
            str(c).title()
            for c in df.columns
        ]

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        if not all(
            c in df.columns
            for c in required
        ):
            return pd.DataFrame()

        return df[required].dropna()

    except Exception:
        return pd.DataFrame()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_indicators(df):
    data = df.copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    data["SMA20"] = (
        close.rolling(20).mean()
    )

    data["SMA50"] = (
        close.rolling(50).mean()
    )

    data["SMA200"] = (
        close.rolling(200).mean()
    )

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    data["MACD"] = (
        ema12 - ema26
    )

    data["MACD_SIGNAL"] = (
        data["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain.rolling(14).mean()
    )

    avg_loss = (
        loss.rolling(14).mean()
    )

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    data["RSI"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    true_range = pd.concat(
        [
            high - low,
            (
                high
                - close.shift()
            ).abs(),
            (
                low
                - close.shift()
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["ATR"] = (
        true_range
        .rolling(14)
        .mean()
    )

    data["VOL20"] = (
        volume.rolling(20).mean()
    )

    data["RETURN5"] = (
        close.pct_change(5)
    )

    data["RETURN20"] = (
        close.pct_change(20)
    )

    return data


# ============================================================
# NEWS
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
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
                "User-Agent": "Mozilla/5.0"
            },
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
                        "title": (
                            html.unescape(
                                title
                            )
                        ),
                        "link": link,
                        "date": date,
                    }
                )

        return articles

    except Exception:
        return []


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
    "outperform",
    "deal",
    "contract",
    "guidance",
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
    "underperform",
    "fraud",
    "penalty",
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
        "sentiment": sentiment,
    }


# ============================================================
# TECHNICAL SCORE
# ============================================================

def technical_score(data):
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
        50,
    )

    macd = safe_float(
        last["MACD"]
    )

    macd_signal = safe_float(
        last["MACD_SIGNAL"]
    )

    score = 0

    score += (
        2
        if close > sma20
        else -2
    )

    score += (
        2
        if close > sma50
        else -2
    )

    score += (
        2
        if sma50 > sma200
        else -2
    )

    score += (
        2
        if macd > macd_signal
        else -2
    )

    if 50 <= rsi <= 70:
        score += 1

    elif rsi < 30:
        score += 1

    elif rsi > 75:
        score -= 1

    return score


# ============================================================
# MACHINE LEARNING PREDICTION ENGINE
# ============================================================

def predict_prices(data):
    if len(data) < 180:
        return None

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
        "RETURN20",
    ]

    df = data.copy()

    df["TARGET1"] = (
        df["Close"].shift(-1)
    )

    df["TARGET5"] = (
        df["Close"].shift(-5)
    )

    df["TARGET20"] = (
        df["Close"].shift(-20)
    )

    clean = df.dropna(
        subset=features
        + [
            "TARGET1",
            "TARGET5",
            "TARGET20",
        ]
    )

    if len(clean) < 100:
        return None

    latest = (
        df[features]
        .iloc[[-1]]
        .dropna()
    )

    if latest.empty:
        return None

    models = {}

    for (
        name,
        target,
        seed,
    ) in [
        ("1d", "TARGET1", 42),
        ("5d", "TARGET5", 43),
        ("20d", "TARGET20", 44),
    ]:

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(
            clean[features],
            clean[target],
        )

        models[name] = model

    return {
        name: safe_float(
            model.predict(
                latest
            )[0]
        )
        for name, model in models.items()
    }


# ============================================================
# NIFTY MARKET TREND
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def market_trend():
    nifty = download_data(
        "^NSEI"
    )

    if nifty.empty:
        return "Unknown", 0

    nifty = add_indicators(
        nifty
    )

    last = nifty.iloc[-1]

    close = safe_float(
        last["Close"]
    )

    sma20 = safe_float(
        last["SMA20"]
    )

    sma50 = safe_float(
        last["SMA50"]
    )

    score = (
        (
            1
            if close > sma20
            else -1
        )
        + (
            1
            if close > sma50
            else -1
        )
    )

    if score == 2:
        return "Bullish", 2

    if score == -2:
        return "Bearish", -2

    return "Neutral", score


# ============================================================
# BUY / SELL / HOLD SIGNAL
# ============================================================

def generate_signal(
    tech_score_value,
    news_score,
    predicted_change,
    market_score_value,
):
    combined = (
        tech_score_value * 0.55
        + news_score * 0.20
        + (
            np.sign(
                predicted_change
            )
            * min(
                abs(predicted_change),
                5,
            )
            * 0.25
        )
        + market_score_value * 0.50
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
# EACH HORIZON GETS ITS OWN LEVELS
# ============================================================

def trade_levels(
    current,
    prediction,
    atr,
    signal,
    horizon_multiplier=1.0,
):
    current = safe_float(
        current
    )

    prediction = safe_float(
        prediction
    )

    atr = safe_float(
        atr
    )

    if current <= 0:
        return {
            "target1": 0,
            "target2": 0,
            "stop": 0,
            "risk": 0,
            "reward": 0,
            "rr": 0,
        }

    if atr <= 0:
        atr = current * 0.02

    atr_scaled = (
        atr
        * max(
            horizon_multiplier,
            1.0,
        )
    )

    if signal in [
        "BUY",
        "STRONG BUY",
    ]:

        target1 = max(
            prediction,
            current
            + atr_scaled * 0.50,
        )

        target2 = max(
            prediction * 1.015,
            current
            + atr_scaled * 1.00,
        )

        stop = (
            current
            - atr_scaled * 1.00
        )

        risk = max(
            current - stop,
            0.01,
        )

        reward = max(
            target1 - current,
            0,
        )

    elif signal in [
        "SELL",
        "STRONG SELL",
    ]:

        target1 = min(
            prediction,
            current
            - atr_scaled * 0.50,
        )

        target2 = min(
            prediction * 0.985,
            current
            - atr_scaled * 1.00,
        )

        stop = (
            current
            + atr_scaled * 1.00
        )

        risk = max(
            stop - current,
            0.01,
        )

        reward = max(
            current - target1,
            0,
        )

    else:
        target1 = prediction
        target2 = prediction

        stop = (
            current - atr_scaled
            if prediction >= current
            else current + atr_scaled
        )

        risk = abs(
            current - stop
        )

        reward = abs(
            target1 - current
        )

    return {
        "target1": max(
            target1,
            0.01,
        ),
        "target2": max(
            target2,
            0.01,
        ),
        "stop": max(
            stop,
            0.01,
        ),
        "risk": risk,
        "reward": reward,
        "rr": (
            reward / risk
            if risk > 0
            else 0
        ),
    }


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def confidence(
    tech_score_value,
    news_score,
    rr,
    predicted_change,
    market_score_value,
):
    score = (
        50
        + min(
            abs(
                tech_score_value
            ) * 4,
            20,
        )
        + min(
            abs(news_score) * 2,
            10,
        )
        + abs(
            market_score_value
        ) * 4
    )

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
        min(score, 92),
    )

    if score >= 75:
        level = "High"

    elif score >= 55:
        level = "Moderate"

    else:
        level = "Low"

    return score, level


# ============================================================
# REASON FOR PREDICTION
# ============================================================

def prediction_reason(
    data,
    news_info,
    market_name,
    predicted_change,
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
        50,
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

    factors.append(
        "price is above the 20-day moving average"
        if close > sma20
        else "price is below the 20-day moving average"
    )

    factors.append(
        "price is above the 50-day moving average"
        if close > sma50
        else "price is below the 50-day moving average"
    )

    factors.append(
        "the medium-term trend is bullish"
        if sma50 > sma200
        else "the medium-term trend remains weak"
    )

    factors.append(
        "MACD is above its signal line"
        if macd > macd_signal
        else "MACD is below its signal line"
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
            "recent news sentiment is neutral or mixed"
        )

    factors.append(
        f"the broader Nifty trend is "
        f"{market_name.lower()}"
    )

    direction = (
        "upside"
        if predicted_change >= 0
        else "downside"
    )

    return (
        f"The model projects approximately "
        f"{abs(predicted_change):.2f}% "
        f"{direction}. "
        + "; ".join(
            factors[:7]
        )
        + "."
    )


# ============================================================
# COMPLETE STOCK ANALYSIS
# ============================================================

def analyze_stock(
    symbol,
    ticker,
    market_name,
    market_score_value,
):
    # Historical data is ONLY used by the prediction engine.
    raw = download_data(ticker)

    if raw.empty:
        return None

    data = add_indicators(
        raw
    ).dropna(
        subset=["Close"]
    )

    if len(data) < 60:
        return None

    last = data.iloc[-1]

    model_base = safe_float(
        last["Close"]
    )

    previous_daily_close = (
        safe_float(
            data["Close"].iloc[-2]
        )
        if len(data) >= 2
        else model_base
    )

    daily_change = (
        (
            model_base
            / previous_daily_close
        ) - 1
    ) * 100 if previous_daily_close else 0

    # ========================================================
    # LIVE QUOTE
    # SEPARATE FROM ML MODEL
    # ========================================================

    live = fetch_live_quote(
        ticker
    )

    live_price = live.get(
        "price"
    )

    live_current = (
        model_base
        if live_price is None
        else safe_float(
            live_price,
            model_base,
        )
    )

    live_previous_close = (
        live.get(
            "previous_close"
        )
    )

    live_change = live.get(
        "change_pct"
    )

    if (
        live_change is None
        and live_previous_close
        and safe_float(
            live_previous_close
        ) > 0
    ):
        live_change = (
            (
                live_current
                / safe_float(
                    live_previous_close
                )
            ) - 1
        ) * 100

    if live_change is None:
        live_change = daily_change

    # ========================================================
    # MODEL INPUTS
    # ========================================================

    tech_score_value = technical_score(
        data
    )

    news = fetch_news(
        symbol
    )

    news_info = news_sentiment(
        news
    )

    ml = predict_prices(
        data
    )

    if ml:
        model_prediction1 = ml["1d"]
        model_prediction5 = ml["5d"]
        model_prediction20 = ml["20d"]

    else:
        model_prediction1 = (
            model_base * 1.005
        )

        model_prediction5 = (
            model_base * 1.015
        )

        model_prediction20 = (
            model_base * 1.03
        )

    model_changes = {
        "1d": (
            (
                model_prediction1
                / model_base
            ) - 1
        ) * 100
        if model_base
        else 0,

        "5d": (
            (
                model_prediction5
                / model_base
            ) - 1
        ) * 100
        if model_base
        else 0,

        "20d": (
            (
                model_prediction20
                / model_base
            ) - 1
        ) * 100
        if model_base
        else 0,
    }

    # ========================================================
    # APPLY MODEL MOVEMENT TO LIVE CMP
    # ========================================================

    predictions = {
        "1d": live_current * (
            1
            + model_changes["1d"]
            / 100
        ),

        "5d": live_current * (
            1
            + model_changes["5d"]
            / 100
        ),

        "20d": live_current * (
            1
            + model_changes["20d"]
            / 100
        ),
    }

    # ========================================================
    # SIGNAL
    # ========================================================

    signal = generate_signal(
        tech_score_value,
        news_info["score"],
        model_changes["1d"],
        market_score_value,
    )

    atr = safe_float(
        last["ATR"]
    )

    # ========================================================
    # THREE INDEPENDENT TRADE PLANS
    # ========================================================

    levels = {
        "1d": trade_levels(
            live_current,
            predictions["1d"],
            atr,
            signal,
            1.0,
        ),

        "5d": trade_levels(
            live_current,
            predictions["5d"],
            atr,
            signal,
            1.5,
        ),

        "20d": trade_levels(
            live_current,
            predictions["20d"],
            atr,
            signal,
            2.0,
        ),
    }

    # ========================================================
    # CONFIDENCE
    # ========================================================

    conf, confidence_level = confidence(
        tech_score_value,
        news_info["score"],
        levels["1d"]["rr"],
        model_changes["1d"],
        market_score_value,
    )

    # ========================================================
    # REASONS
    # ========================================================

    reasons = {
        "1d": prediction_reason(
            data,
            news_info,
            market_name,
            model_changes["1d"],
        ),

        "5d": prediction_reason(
            data,
            news_info,
            market_name,
            model_changes["5d"],
        ),

        "20d": prediction_reason(
            data,
            news_info,
            market_name,
            model_changes["20d"],
        ),
    }

    return {
        "symbol": symbol,
        "ticker": ticker,
        "data": data,
        "model_base": model_base,
        "current": live_current,
        "today_change": live_change,
        "daily_change": daily_change,
        "signal": signal,
        "tech_score": tech_score_value,
        "news": news,
        "news_info": news_info,
        "predictions": predictions,
        "changes": model_changes,
        "levels": levels,
        "confidence": conf,
        "confidence_level": confidence_level,
        "reasons": reasons,
        "market_name": market_name,
        "market_score": market_score_value,
        "live_quote": live,
        "market_date": pd.Timestamp(
            data.index[-1]
        ).strftime("%d %b %Y"),
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
        "Custom Stock",
    ],
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
            "TCS",
        )
        .upper()
        .strip()
    )

    selected_ticker = (
        selected_symbol
        if selected_symbol.endswith(".NS")
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
                if "TCS"
                in universe
                else 0
            ),
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
    use_container_width=True,
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
        Technical analysis • Machine learning •
        News sentiment • Market trend •
        Buy/Sell signals • Risk management
    </div>
    """,
    unsafe_allow_html=True,
)

market_name, market_score_value = (
    market_trend()
)


# ============================================================
# NIFTY SCANNER
# ============================================================

if scan:

    st.markdown(
        f"""
        <div class="section-title">
            🔎 {html.escape(universe_name)} Scanner
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = []

    progress = st.progress(
        0
    )

    symbols = list(
        universe.items()
    )

    for index, (
        symbol,
        ticker,
    ) in enumerate(symbols):

        try:
            result = analyze_stock(
                symbol,
                ticker,
                market_name,
                market_score_value,
            )

            if result:

                rows.append(
                    {
                        "Stock": symbol,

                        "CMP": round(
                            result[
                                "current"
                            ],
                            2,
                        ),

                        "Signal": result[
                            "signal"
                        ],

                        "1D Target": round(
                            result[
                                "predictions"
                            ]["1d"],
                            2,
                        ),

                        "1D %": round(
                            result[
                                "changes"
                            ]["1d"],
                            2,
                        ),

                        "5D %": round(
                            result[
                                "changes"
                            ]["5d"],
                            2,
                        ),

                        "20D %": round(
                            result[
                                "changes"
                            ]["20d"],
                            2,
                        ),

                        "1D Stop Loss": round(
                            result[
                                "levels"
                            ]["1d"]["stop"],
                            2,
                        ),

                        "1D R/R": (
                            "1 : "
                            f"{result['levels']['1d']['rr']:.2f}"
                        ),

                        "Confidence": (
                            f"{result['confidence']:.0f}%"
                        ),

                        "News": result[
                            "news_info"
                        ]["sentiment"],

                        "Market Status": result[
                            "live_quote"
                        ]["market_status"],
                    }
                )

        except Exception:
            pass

        progress.progress(
            (index + 1)
            / max(
                len(symbols),
                1,
            )
        )

    progress.empty()

    if rows:

        scan_df = pd.DataFrame(
            rows
        )

        signal_order = {
            "STRONG BUY": 0,
            "BUY": 1,
            "HOLD": 2,
            "SELL": 3,
            "STRONG SELL": 4,
        }

        scan_df["_order"] = (
            scan_df[
                "Signal"
            ]
            .map(
                signal_order
            )
            .fillna(5)
        )

        scan_df = (
            scan_df
            .sort_values(
                [
                    "_order",
                    "Confidence",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .drop(
                columns=[
                    "_order"
                ]
            )
        )

        st.dataframe(
            scan_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Download Scanner",
            data=scan_df.to_csv(
                index=False
            ),
            file_name=(
                "stock_scanner.csv"
            ),
            mime="text/csv",
        )

    else:

        st.warning(
            "No stock data could be retrieved."
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
        selected_ticker,
        market_name,
        market_score_value,
    )


if result is None:

    st.error(
        f"""
        Unable to retrieve enough market data
        for **{selected_symbol}**.

        Please check the stock symbol,
        internet connection, and Yahoo Finance
        availability.
        """
    )

    st.stop()


data = result["data"]

last = data.iloc[-1]

current = result["current"]

today_change = result[
    "today_change"
]

signal = result["signal"]

levels = result["levels"]

news = result["news"]

news_info = result[
    "news_info"
]

predictions = result[
    "predictions"
]

changes = result[
    "changes"
]

live_quote = result[
    "live_quote"
]

quote_time = live_quote.get(
    "quote_time"
)

quote_time_text = (
    quote_time.strftime(
        "%d %b %Y, %I:%M:%S %p IST"
    )
    if quote_time
    else "Unavailable"
)

market_status = live_quote.get(
    "market_status",
    "Unavailable",
)

status_detail = live_quote.get(
    "status_detail",
    "Live quote unavailable",
)


if market_status == "Market Open":

    cmp_label = (
        "Live CMP / Last Traded Price"
    )

    cmp_note = (
        "Live market quote"
    )

    status_class = (
        "status-open"
    )

elif market_status == "Market Closed":

    cmp_label = (
        "Last Traded Price"
    )

    cmp_note = (
        status_detail
    )

    status_class = (
        "status-closed"
    )

else:

    cmp_label = (
        "Latest Available Price"
    )

    cmp_note = (
        "Live quote unavailable"
    )

    status_class = (
        "status-unavailable"
    )


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


# ============================================================
# LIVE CMP / MARKET STATUS
# ============================================================

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
            {cmp_label}
        </div>

        <div class="cmp-price">
            {money(current)}
        </div>

        <div class="{price_class}">
            {arrow} {abs(today_change):.2f}% today
        </div>

        <div style="margin-top:12px;">
            <span class="status-pill {status_class}">
                ● {html.escape(market_status)}
            </span>
        </div>

        <div style="
            margin-top:8px;
            color:#64748b;
            font-size:11px;
        ">
            {html.escape(cmp_note)}
        </div>

        <div style="
            margin-top:6px;
            color:#64748b;
            font-size:11px;
        ">
            Quote time:
            {html.escape(quote_time_text)}
        </div>

        <div style="
            margin-top:4px;
            color:#64748b;
            font-size:11px;
        ">
            Prediction model data through:
            {html.escape(result["market_date"])}
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SUMMARY METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(
    4
)

with c1:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Trading Signal
            </div>

            <div class="
                info-value
                {signal_class(signal)}
            ">
                {signal}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with c2:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Confidence
            </div>

            <div class="info-value">
                {result["confidence"]:.0f}%
            </div>

            <div class="small-text">
                {result["confidence_level"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with c3:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Technical Score
            </div>

            <div class="info-value">
                {result["tech_score"]}
            </div>

            <div class="small-text">
                Approx. range -9 to +9
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with c4:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Market Trend
            </div>

            <div class="info-value">
                {html.escape(market_name)}
            </div>

            <div class="small-text">
                Nifty trend
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# AI PRICE PREDICTIONS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🤖 AI Price Predictions
    </div>
    """,
    unsafe_allow_html=True,
)

prediction_items = [
    (
        "1 Trading Day",
        "1d",
        1,
    ),
    (
        "5 Trading Days",
        "5d",
        5,
    ),
    (
        "20 Trading Days",
        "20d",
        20,
    ),
]

prediction_cols = st.columns(
    3
)

for index, (
    horizon,
    key,
    days,
) in enumerate(
    prediction_items
):

    predicted = predictions[
        key
    ]

    change = changes[
        key
    ]

    level = levels[
        key
    ]

    cls = (
        "positive"
        if change >= 0
        else "negative"
    )

    direction_arrow = (
        "↑"
        if change >= 0
        else "↓"
    )

    with prediction_cols[index]:

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-heading">
                    {horizon}
                </div>

                <div class="small-text"
                     style="margin-top:8px;">
                    Model projected price
                </div>

                <div class="prediction-price">
                    {money(predicted)}
                </div>

                <div class="{cls}"
                     style="margin-top:5px;">
                    {direction_arrow}
                    {abs(change):.2f}%
                </div>

                <div style="margin-top:22px;">

                    <div class="small-text">
                        Target 1
                    </div>

                    <div style="
                        color:#22c55e;
                        font-size:16px;
                        font-weight:700;
                    ">
                        {money(level["target1"])}
                    </div>

                    <div class="small-text"
                         style="margin-top:12px;">
                        Target 2
                    </div>

                    <div style="
                        color:#22c55e;
                        font-size:16px;
                        font-weight:700;
                    ">
                        {money(level["target2"])}
                    </div>

                    <div class="small-text"
                         style="margin-top:12px;">
                        Stop Loss
                    </div>

                    <div style="
                        color:#ef4444;
                        font-size:16px;
                        font-weight:700;
                    ">
                        {money(level["stop"])}
                    </div>

                    <div class="small-text"
                         style="margin-top:12px;">
                        Risk / Reward
                    </div>

                    <div style="
                        color:#60a5fa;
                        font-size:16px;
                        font-weight:700;
                    ">
                        1 : {level["rr"]:.2f}
                    </div>

                    <div class="small-text"
                         style="margin-top:12px;">
                        Estimated Date
                    </div>

                    <div style="
                        color:#f8fafc;
                        font-size:13px;
                    ">
                        {business_day_date(days)}
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# REASON FOR EACH PREDICTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🧠 Why does the model expect this movement?
    </div>
    """,
    unsafe_allow_html=True,
)

reason_cols = st.columns(
    3
)

for index, (
    horizon,
    key,
    days,
) in enumerate(
    prediction_items
):

    with reason_cols[index]:

        st.markdown(
            f"""
            <div class="reason-box">

                <div class="reason-title">
                    {horizon} — Prediction Reason
                </div>

                <div class="reason-text">
                    {html.escape(
                        result["reasons"][key]
                    )}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
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
    unsafe_allow_html=True,
)

t1, t2, t3, t4, t5 = (
    st.columns(5)
)

with t1:

    st.metric(
        "Entry / CMP",
        money(current),
    )

with t2:

    st.metric(
        "1D Target",
        money(
            levels["1d"][
                "target1"
            ]
        ),
    )

with t3:

    st.metric(
        "5D Target",
        money(
            levels["5d"][
                "target1"
            ]
        ),
    )

with t4:

    st.metric(
        "20D Target",
        money(
            levels["20d"][
                "target1"
            ]
        ),
    )

with t5:

    st.metric(
        "1D Stop Loss",
        money(
            levels["1d"][
                "stop"
            ]
        ),
    )


# ============================================================
# RISK / REWARD BY HORIZON
# ============================================================

st.markdown(
    """
    <div class="section-title">
        ⚖️ Risk / Reward by Horizon
    </div>
    """,
    unsafe_allow_html=True,
)

r1, r2, r3 = st.columns(
    3
)

with r1:

    st.metric(
        "1 Trading Day",
        (
            "1 : "
            f"{levels['1d']['rr']:.2f}"
        ),
    )

with r2:

    st.metric(
        "5 Trading Days",
        (
            "1 : "
            f"{levels['5d']['rr']:.2f}"
        ),
    )

with r3:

    st.metric(
        "20 Trading Days",
        (
            "1 : "
            f"{levels['20d']['rr']:.2f}"
        ),
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
    unsafe_allow_html=True,
)

a, b, c, d, e, f = (
    st.columns(6)
)

with a:

    st.metric(
        "RSI",
        f"{safe_float(last['RSI'], 50):.1f}",
    )

with b:

    st.metric(
        "20D MA",
        money(last["SMA20"]),
    )

with c:

    st.metric(
        "50D MA",
        money(last["SMA50"]),
    )

with d:

    st.metric(
        "200D MA",
        money(last["SMA200"]),
    )

with e:

    st.metric(
        "MACD",
        f"{safe_float(last['MACD']):.2f}",
    )

with f:

    st.metric(
        "ATR",
        money(last["ATR"]),
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
    unsafe_allow_html=True,
)

if news:

    for article in news[:6]:

        title = html.escape(
            article["title"]
        )

        link = html.escape(
            article["link"],
            quote=True,
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
            unsafe_allow_html=True,
        )

else:

    st.info(
        "No recent news was retrieved."
    )


# ============================================================
# KEY MOVEMENT FACTORS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📌 Key Movement Factors
    </div>
    """,
    unsafe_allow_html=True,
)

factors = []

factors.append(
    (
        "Price is above the 20-day "
        "moving average."
        if current
        > safe_float(
            last["SMA20"]
        )
        else
        "Price is below the 20-day "
        "moving average."
    )
)

factors.append(
    (
        "Price is above the 50-day "
        "moving average."
        if current
        > safe_float(
            last["SMA50"]
        )
        else
        "Price is below the 50-day "
        "moving average."
    )
)

factors.append(
    (
        "MACD is supporting bullish "
        "momentum."
        if safe_float(
            last["MACD"]
        )
        > safe_float(
            last["MACD_SIGNAL"]
        )
        else
        "MACD is showing bearish "
        "momentum."
    )
)

rsi = safe_float(
    last["RSI"],
    50,
)

if rsi < 30:

    factors.append(
        "RSI is oversold and a "
        "technical rebound is possible."
    )

elif rsi > 70:

    factors.append(
        "RSI is overbought and "
        "short-term cooling is possible."
    )

else:

    factors.append(
        f"RSI is currently {rsi:.1f}."
    )


if safe_float(
    last["Volume"]
) > safe_float(
    last["VOL20"]
):

    factors.append(
        "Trading volume is above "
        "its 20-day average."
    )


if news_info[
    "sentiment"
] == "Positive":

    factors.append(
        "Recent news sentiment is positive."
    )

elif news_info[
    "sentiment"
] == "Negative":

    factors.append(
        "Recent news sentiment is negative."
    )

else:

    factors.append(
        "Recent news sentiment is neutral "
        "or mixed."
    )


factors.append(
    f"Nifty market trend is currently "
    f"{market_name}."
)


for factor in factors:

    st.markdown(
        f"""
        <div class="factor">
            • {html.escape(factor)}
        </div>
        """,
        unsafe_allow_html=True,
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

        This application uses market data,
        technical indicators, recent news sentiment
        and machine-learning models to generate
        estimates.

        Predictions are not guaranteed and are not
        financial advice. Actual prices can differ
        materially from model estimates.

        Always perform your own research and consider
        your risk tolerance.

    </div>
    """,
    unsafe_allow_html=True,
)
