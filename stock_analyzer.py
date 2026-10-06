import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import time

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

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0d1728 50%, #07111f 100%);
    color: #f8fafc;
}

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

/* Main title */
.dashboard-title {
    background: linear-gradient(135deg, #111827, #182338);
    border: 1px solid #26344d;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
}

.dashboard-title h1 {
    margin: 0;
    color: #f8fafc;
    font-size: 38px;
    font-weight: 800;
}

.dashboard-title p {
    margin-top: 8px;
    color: #94a3b8;
    font-size: 16px;
}

/* Cards */
.metric-card {
    background: linear-gradient(145deg, #111827, #151f31);
    border: 1px solid #26344d;
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.20);
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 8px;
}

.metric-value {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 750;
}

.metric-sub {
    color: #64748b;
    font-size: 12px;
    margin-top: 7px;
}

/* Signal cards */
.signal-card {
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    border: 1px solid #334155;
    margin: 8px 0;
}

.signal-buy {
    background: linear-gradient(135deg, #062e24, #064e3b);
    border-color: #10b981;
}

.signal-sell {
    background: linear-gradient(135deg, #3b1010, #5b1515);
    border-color: #ef4444;
}

.signal-hold {
    background: linear-gradient(135deg, #33260a, #4a3410);
    border-color: #f59e0b;
}

.signal-title {
    color: white;
    font-size: 31px;
    font-weight: 800;
}

.signal-score {
    color: #cbd5e1;
    font-size: 14px;
    margin-top: 5px;
}

/* Section */
.section-title {
    color: #f8fafc;
    font-size: 23px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* Info */
.info-box {
    background: #111827;
    border: 1px solid #26344d;
    border-radius: 14px;
    padding: 18px;
    color: #cbd5e1;
    margin: 10px 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 35px 0 10px 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="dashboard-title">
    <h1>📈 Indian Stock AI Analyzer</h1>
    <p>
        Live market view • Technical analysis • Buy/Sell signals •
        Multi-horizon prediction • News trend analysis
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STOCK LISTS
# ============================================================

NIFTY_TOP_10 = [
    "RELIANCE",
    "HDFCBANK",
    "TCS",
    "BHARTIARTL",
    "ICICIBANK",
    "INFY",
    "SBIN",
    "HINDUNILVR",
    "ITC",
    "LT"
]

NIFTY_50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT",
    "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BEL", "BPCL", "BHARTIARTL", "BRITANNIA",
    "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "MAXHEALTH",
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO",
    "WIPRO"
]

# Approximate extended Nifty 100 universe
NIFTY_100_EXTRA = [
    "ABB", "ACC", "AMBUJACEM", "AUROPHARMA", "BANKBARODA",
    "BHEL", "BOSCHLTD", "CANBK", "COLPAL", "CONCOR",
    "CUMMINSIND", "DABUR", "DLF", "GAIL", "GODREJCP",
    "GODREJPROP", "HAL", "HAVELLS", "ICICIPRULI",
    "INDHOTEL", "INDUSTOWER", "IOC", "IRCTC", "JINDALSTEL",
    "LICI", "LUPIN", "MARICO", "MOTHERSON", "NAUKRI",
    "NMDC", "OFSS", "PAGEIND", "PFC", "PIDILITIND",
    "PNB", "RECLTD", "SAIL", "SRF", "TORNTPHARM",
    "TVSMOTOR", "VEDL", "VOLTAS", "YESBANK"
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Analyzer")

    market = st.selectbox(
        "Market",
        ["NSE", "BSE"]
    )

    symbol = st.text_input(
        "Stock Symbol",
        value="TCS",
        placeholder="Example: TCS"
    ).strip().upper()

    st.caption("Examples: TCS, RELIANCE, INFY, HDFCBANK")

    st.markdown("---")

    st.markdown("### 📊 Scanner")

    scanner = st.selectbox(
        "Select Universe",
        [
            "Single Stock",
            "Nifty Top 10",
            "Nifty 50",
            "Nifty 100"
        ]
    )

    st.markdown("---")

    analysis_period = st.selectbox(
        "Analysis Period",
        [
            "6 Months",
            "1 Year",
            "2 Years"
        ],
        index=0
    )

    news_count = st.slider(
        "News Articles",
        min_value=3,
        max_value=10,
        value=5
    )

    st.markdown("---")

    refresh = st.button(
        "🔄 Refresh Market Data",
        use_container_width=True
    )

    st.caption(
        "Market prices may be delayed depending on the data provider."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_ticker(symbol, market):
    """
    Convert Indian stock symbol to Yahoo Finance ticker.
    """
    if market == "NSE":
        return f"{symbol}.NS"
    return f"{symbol}.BO"


def flatten_yfinance_columns(df):
    """
    Fix yfinance MultiIndex columns.
    This is important because newer yfinance versions
    can return MultiIndex columns even for a single ticker.
    """

    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):

        # For one ticker, use the first level where possible.
        if df.columns.nlevels >= 2:

            try:
                df.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df.columns
                ]
            except Exception:
                df.columns = [
                    str(col[0]) if isinstance(col, tuple) else str(col)
                    for col in df.columns
                ]

    df.columns = [str(c) for c in df.columns]

    return df


def clean_price_data(df):

    if df is None or df.empty:
        return pd.DataFrame()

    df = flatten_yfinance_columns(df)

    required = ["Open", "High", "Low", "Close", "Volume"]

    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    return df


def calculate_indicators(data):

    df = data.copy()

    close = df["Close"]

    # Moving averages
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_Hist"] = (
        df["MACD"] - df["Signal"]
    )

    # Bollinger Bands
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    df["BB_Middle"] = middle
    df["BB_Upper"] = middle + 2 * std
    df["BB_Lower"] = middle - 2 * std

    # Returns
    df["Return_1D"] = close.pct_change(1) * 100
    df["Return_5D"] = close.pct_change(5) * 100
    df["Return_20D"] = close.pct_change(20) * 100

    # Volatility
    df["Volatility"] = (
        df["Return_1D"].rolling(20).std()
    )

    # Average volume
    df["Volume_MA20"] = (
        df["Volume"].rolling(20).mean()
    )

    return df


def get_latest_price(ticker_obj, data):

    """
    Try Yahoo fast_info first.
    Fall back to the latest downloaded close.
    """

    live_price = None

    try:
        fast_info = ticker_obj.fast_info

        live_price = fast_info.get(
            "last_price"
        )

        if live_price is not None:
            live_price = float(live_price)

    except Exception:
        live_price = None

    if live_price is None or not np.isfinite(live_price):

        if not data.empty:
            live_price = float(
                data["Close"].iloc[-1]
            )

    return live_price


def get_news(ticker_obj, count=5):

    """
    Retrieve recent Yahoo Finance news.
    """

    articles = []

    try:
        raw_news = ticker_obj.news

        if not raw_news:
            return articles

        for item in raw_news[:count]:

            title = ""
            publisher = ""
            link = ""
            timestamp = None

            # New Yahoo structure
            content = item.get("content", {})

            if isinstance(content, dict):

                title = content.get(
                    "title",
                    ""
                )

                provider = content.get(
                    "provider",
                    {}
                )

                if isinstance(provider, dict):
                    publisher = provider.get(
                        "displayName",
                        ""
                    )

                canonical = content.get(
                    "canonicalUrl",
                    {}
                )

                if isinstance(canonical, dict):
                    link = canonical.get(
                        "url",
                        ""
                    )

                pub_date = content.get(
                    "pubDate"
                )

                if pub_date:
                    timestamp = pub_date

            # Older Yahoo structure
            if not title:
                title = item.get(
                    "title",
                    ""
                )

            if not publisher:
                publisher = item.get(
                    "publisher",
                    ""
                )

            if not link:
                link = item.get(
                    "link",
                    ""
                )

            if title:
                articles.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "timestamp": timestamp
                })

    except Exception:
        pass

    return articles


def news_sentiment(articles):

    """
    Simple transparent news sentiment model.
    This is NOT a financial prediction model.
    """

    positive_words = [
        "profit",
        "growth",
        "surge",
        "strong",
        "upgrade",
        "buy",
        "bullish",
        "record",
        "positive",
        "beat",
        "outperform",
        "partnership",
        "deal",
        "expansion",
        "revenue",
        "dividend"
    ]

    negative_words = [
        "loss",
        "fall",
        "drop",
        "weak",
        "downgrade",
        "sell",
        "bearish",
        "decline",
        "negative",
        "miss",
        "fraud",
        "investigation",
        "debt",
        "risk",
        "warning",
        "lawsuit"
    ]

    if not articles:
        return 0, "Neutral"

    score = 0

    for article in articles:

        title = article["title"].lower()

        for word in positive_words:
            if word in title:
                score += 1

        for word in negative_words:
            if word in title:
                score -= 1

    if score >= 3:
        sentiment = "Positive"

    elif score <= -3:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return score, sentiment


def calculate_signal(df, news_score):

    """
    Combine technical indicators + news sentiment
    into a transparent score.
    """

    latest = df.iloc[-1]

    score = 0
    reasons = []

    price = float(latest["Close"])

    # Price vs MA20
    if pd.notna(latest["MA20"]):

        if price > latest["MA20"]:
            score += 1
            reasons.append(
                "Price is above MA20"
            )
        else:
            score -= 1
            reasons.append(
                "Price is below MA20"
            )

    # MA20 vs MA50
    if (
        pd.notna(latest["MA20"])
        and pd.notna(latest["MA50"])
    ):

        if latest["MA20"] > latest["MA50"]:
            score += 2
            reasons.append(
                "MA20 is above MA50"
            )
        else:
            score -= 2
            reasons.append(
                "MA20 is below MA50"
            )

    # RSI
    rsi = latest["RSI"]

    if pd.notna(rsi):

        if 50 <= rsi <= 70:
            score += 2
            reasons.append(
                f"RSI is healthy at {rsi:.1f}"
            )

        elif rsi < 30:
            score += 2
            reasons.append(
                f"RSI indicates oversold conditions ({rsi:.1f})"
            )

        elif rsi > 75:
            score -= 2
            reasons.append(
                f"RSI indicates overbought conditions ({rsi:.1f})"
            )

        elif rsi < 45:
            score -= 1

    # MACD
    if (
        pd.notna(latest["MACD"])
        and pd.notna(latest["Signal"])
    ):

        if latest["MACD"] > latest["Signal"]:
            score += 2
            reasons.append(
                "MACD is bullish"
            )
        else:
            score -= 2
            reasons.append(
                "MACD is bearish"
            )

    # Bollinger
    if pd.notna(latest["BB_Middle"]):

        if price > latest["BB_Middle"]:
            score += 1
        else:
            score -= 1

    # News
    if news_score >= 3:
        score += 2
        reasons.append(
            "Recent news trend is positive"
        )

    elif news_score <= -3:
        score -= 2
        reasons.append(
            "Recent news trend is negative"
        )

    # Final signal
    if score >= 4:
        signal = "BUY"
    elif score <= -4:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, score, reasons


def calculate_prediction(df, news_score):

    """
    Multi-horizon directional estimate.

    This is intentionally presented as a probability-style
    directional estimate rather than a guaranteed future price.
    """

    latest = df.iloc[-1]

    current = float(latest["Close"])

    rsi = (
        float(latest["RSI"])
        if pd.notna(latest["RSI"])
        else 50
    )

    macd_hist = (
        float(latest["MACD_Hist"])
        if pd.notna(latest["MACD_Hist"])
        else 0
    )

    trend_5d = (
        float(df["Return_5D"].iloc[-1])
        if pd.notna(df["Return_5D"].iloc[-1])
        else 0
    )

    trend_20d = (
        float(df["Return_20D"].iloc[-1])
        if pd.notna(df["Return_20D"].iloc[-1])
        else 0
    )

    technical_score = 0

    if pd.notna(latest["MA20"]):
        technical_score += (
            1 if current > latest["MA20"]
            else -1
        )

    if pd.notna(latest["MA50"]):
        technical_score += (
            1 if current > latest["MA50"]
            else -1
        )

    if rsi > 50:
        technical_score += 1
    elif rsi < 45:
        technical_score -= 1

    if macd_hist > 0:
        technical_score += 1
    else:
        technical_score -= 1

    # News influence
    news_component = np.clip(
        news_score / 3,
        -2,
        2
    )

    combined = (
        technical_score + news_component
    )

    # Estimate directional movement.
    # These are scenario estimates, not guarantees.
    daily_move = (
        0.25
        + abs(trend_5d) * 0.03
        + abs(macd_hist / current) * 100 * 0.20
    )

    daily_move = float(
        np.clip(daily_move, 0.20, 1.50)
    )

    direction = 1 if combined >= 0 else -1

    pred_1d = current * (
        1 + direction * daily_move / 100
    )

    pred_5d = current * (
        1 + direction * daily_move * 2.2 / 100
    )

    pred_20d = current * (
        1 + direction * daily_move * 5.0 / 100
    )

    confidence = 50 + (
        min(abs(combined) * 7, 30)
    )

    confidence = min(
        max(confidence, 50),
        80
    )

    if direction > 0:
        outlook = "Bullish"
    else:
        outlook = "Bearish"

    return {
        "1D": pred_1d,
        "5D": pred_5d,
        "20D": pred_20d,
        "confidence": confidence,
        "outlook": outlook
    }


def format_currency(value):

    if value is None:
        return "N/A"

    return f"₹{value:,.2f}"


def signal_class(signal):

    if signal == "BUY":
        return "signal-buy"

    if signal == "SELL":
        return "signal-sell"

    return "signal-hold"


def run_stock_analysis(symbol, market):

    ticker_symbol = get_ticker(
        symbol,
        market
    )

    ticker = yf.Ticker(ticker_symbol)

    # Determine download period
    if analysis_period == "6 Months":
        period = "6mo"

    elif analysis_period == "1 Year":
        period = "1y"

    else:
        period = "2y"

    try:

        data = ticker.history(
            period=period,
            interval="1d",
            auto_adjust=False
        )

    except Exception as e:

        st.error(
            f"Unable to download data for {ticker_symbol}: {e}"
        )

        return None

    data = clean_price_data(data)

    if data.empty:
        st.error(
            f"No market data found for {ticker_symbol}."
        )
        return None

    data = calculate_indicators(data)

    latest_price = get_latest_price(
        ticker,
        data
    )

    if latest_price is None:
        st.error("Unable to determine latest price.")
        return None

    # Replace latest close with current price if available
    analysis_data = data.copy()

    if len(analysis_data) > 0:
        analysis_data.loc[
            analysis_data.index[-1],
            "Close"
        ] = latest_price

    news = get_news(
        ticker,
        news_count
    )

    news_score, news_label = news_sentiment(
        news
    )

    signal, score, reasons = calculate_signal(
        analysis_data,
        news_score
    )

    prediction = calculate_prediction(
        analysis_data,
        news_score
    )

    return {
        "ticker": ticker_symbol,
        "data": analysis_data,
        "price": latest_price,
        "news": news,
        "news_score": news_score,
        "news_label": news_label,
        "signal": signal,
        "score": score,
        "reasons": reasons,
        "prediction": prediction
    }


# ============================================================
# EMPTY STATE
# ============================================================

if not symbol:

    st.markdown("""
    <div style="
        text-align:center;
        padding:65px 30px;
        background:#171a21;
        border-radius:20px;
        margin-top:30px;
        border:1px solid #26344d;
    ">

        <div style="font-size:55px;">📊</div>

        <h2 style="color:#f8fafc;">
            Start Your Analysis
        </h2>

        <p style="
            color:#cbd5e1;
            font-size:18px;
            line-height:1.6;
        ">
            Enter a stock symbol in the sidebar to view
            the latest price and technical analysis.
        </p>

        <p style="color:#64748b;font-size:15px;">
            Examples: TCS • RELIANCE • INFY • HDFCBANK
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# SCANNER MODE
# ============================================================

if scanner != "Single Stock":

    if scanner == "Nifty Top 10":
        stock_list = NIFTY_TOP_10

    elif scanner == "Nifty 50":
        stock_list = NIFTY_50

    else:
        stock_list = list(
            dict.fromkeys(
                NIFTY_50 + NIFTY_100_EXTRA
            )
        )

    st.markdown(
        f'<div class="section-title">📊 {scanner} Scanner</div>',
        unsafe_allow_html=True
    )

    results = []

    progress = st.progress(0)

    for i, stock in enumerate(stock_list):

        try:

            ticker_symbol = get_ticker(
                stock,
                market
            )

            ticker = yf.Ticker(
                ticker_symbol
            )

            data = ticker.history(
                period="6mo",
                interval="1d",
                auto_adjust=False
            )

            data = clean_price_data(data)

            if data.empty:
                continue

            data = calculate_indicators(data)

            news = get_news(
                ticker,
                3
            )

            news_score, news_label = news_sentiment(
                news
            )

            signal, score, reasons = calculate_signal(
                data,
                news_score
            )

            current = float(
                data["Close"].iloc[-1]
            )

            change = float(
                data["Return_1D"].iloc[-1]
            ) if pd.notna(
                data["Return_1D"].iloc[-1]
            ) else 0

            prediction = calculate_prediction(
                data,
                news_score
            )

            results.append({
                "Stock": stock,
                "CMP": current,
                "1D %": change,
                "Signal": signal,
                "Score": score,
                "News": news_label,
                "5D Target": prediction["5D"]
            })

        except Exception:
            pass

        progress.progress(
            int(
                ((i + 1) / len(stock_list)) * 100
            )
        )

    progress.empty()

    if results:

        scan_df = pd.DataFrame(results)

        scan_df = scan_df.sort_values(
            "Score",
            ascending=False
        )

        display_df = scan_df.copy()

        display_df["CMP"] = display_df[
            "CMP"
        ].map(
            lambda x: f"₹{x:,.2f}"
        )

        display_df["5D Target"] = display_df[
            "5D Target"
        ].map(
            lambda x: f"₹{x:,.2f}"
        )

        display_df["1D %"] = display_df[
            "1D %"
        ].map(
            lambda x: f"{x:+.2f}%"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Scanner rankings are based on technical indicators "
            "and recent headline sentiment. They are not guaranteed "
            "investment recommendations."
        )

    else:

        st.warning(
            "No stocks could be analyzed right now."
        )

    st.markdown(
        '<div class="footer">Indian Stock AI Analyzer • Market data powered by Yahoo Finance</div>',
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# SINGLE STOCK ANALYSIS
# ============================================================

with st.spinner(
    f"Analyzing {symbol}..."
):

    result = run_stock_analysis(
        symbol,
        market
    )


if result is None:
    st.stop()


data = result["data"]
price = result["price"]
signal = result["signal"]
score = result["score"]
news = result["news"]
news_label = result["news_label"]
prediction = result["prediction"]


# ============================================================
# TIMESTAMP
# ============================================================

now = datetime.now()

last_market_date = data.index[-1]

if hasattr(
    last_market_date,
    "strftime"
):

    market_date = last_market_date.strftime(
        "%d %b %Y"
    )

else:

    market_date = str(
        last_market_date
    )


st.markdown(
    f"""
    <div style="
        color:#64748b;
        font-size:13px;
        margin-bottom:15px;
    ">
        Data date: <b>{market_date}</b>
        &nbsp; • &nbsp;
        Dashboard updated: <b>{now.strftime("%d %b %Y, %I:%M:%S %p")}</b>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TOP METRICS
# ============================================================

latest = data.iloc[-1]

rsi_value = (
    float(latest["RSI"])
    if pd.notna(latest["RSI"])
    else 0
)

ma20 = (
    float(latest["MA20"])
    if pd.notna(latest["MA20"])
    else 0
)

ma50 = (
    float(latest["MA50"])
    if pd.notna(latest["MA50"])
    else 0
)

daily_change = (
    float(latest["Return_1D"])
    if pd.notna(latest["Return_1D"])
    else 0
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Current Market Price
            </div>
            <div class="metric-value">
                {format_currency(price)}
            </div>
            <div class="metric-sub">
                {result["ticker"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    change_sign = "+" if daily_change >= 0 else ""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Today's Move
            </div>
            <div class="metric-value">
                {change_sign}{daily_change:.2f}%
            </div>
            <div class="metric-sub">
                Previous trading session
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                RSI
            </div>
            <div class="metric-value">
                {rsi_value:.1f}
            </div>
            <div class="metric-sub">
                Momentum indicator
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                MA20
            </div>
            <div class="metric-value">
                {format_currency(ma20)}
            </div>
            <div class="metric-sub">
                Short-term trend
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col5:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                MA50
            </div>
            <div class="metric-value">
                {format_currency(ma50)}
            </div>
            <div class="metric-sub">
                Medium-term trend
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIGNAL
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Trading Signal</div>',
    unsafe_allow_html=True
)

signal_col, explanation_col = st.columns(
    [1, 2]
)


with signal_col:

    st.markdown(
        f"""
        <div class="signal-card {signal_class(signal)}">

            <div class="signal-title">
                {signal}
            </div>

            <div class="signal-score">
                Technical + news score: {score}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with explanation_col:

    st.markdown(
        """
        <div class="info-box">
            <b>Why this signal?</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    for reason in result["reasons"][:6]:
        st.write("•", reason)


# ============================================================
# PREDICTION TIMELINE
# ============================================================

st.markdown(
    '<div class="section-title">🔮 Prediction Timeline</div>',
    unsafe_allow_html=True
)

p1, p2, p3, p4 = st.columns(4)


with p1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Next Trading Day
            </div>
            <div class="metric-value">
                {format_currency(prediction["1D"])}
            </div>
            <div class="metric-sub">
                Direction: {prediction["outlook"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with p2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                ~5 Trading Days
            </div>
            <div class="metric-value">
                {format_currency(prediction["5D"])}
            </div>
            <div class="metric-sub">
                Short-term scenario
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with p3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                ~20 Trading Days
            </div>
            <div class="metric-value">
                {format_currency(prediction["20D"])}
            </div>
            <div class="metric-sub">
                Medium-term scenario
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with p4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Model Confidence
            </div>
            <div class="metric-value">
                {prediction["confidence"]:.0f}%
            </div>
            <div class="metric-sub">
                Directional confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.warning(
    "Prediction values are scenario estimates based on historical technical "
    "patterns and recent headline sentiment. They are not guaranteed future prices."
)


# ============================================================
# PRICE CHART
# ============================================================

st.markdown(
    '<div class="section-title">📊 Price & Trend</div>',
    unsafe_allow_html=True
)

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    data.index,
    data["Close"],
    label="Close Price",
    linewidth=2
)

ax.plot(
    data.index,
    data["MA20"],
    label="MA20",
    linewidth=1.5
)

ax.plot(
    data.index,
    data["MA50"],
    label="MA50",
    linewidth=1.5
)

if data["MA200"].notna().any():

    ax.plot(
        data.index,
        data["MA200"],
        label="MA200",
        linewidth=1.3
    )

ax.set_title(
    f"{symbol} Price Trend"
)

ax.set_ylabel(
    "Price (₹)"
)

ax.grid(
    alpha=0.2
)

ax.legend()

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

technical_left, technical_right = st.columns(2)


with technical_left:

    st.markdown(
        '<div class="section-title">🌀 RSI</div>',
        unsafe_allow_html=True
    )

    fig_rsi, ax_rsi = plt.subplots(
        figsize=(8, 3)
    )

    ax_rsi.plot(
        data.index,
        data["RSI"],
        linewidth=1.5
    )

    ax_rsi.axhline(
        70,
        linestyle="--"
    )

    ax_rsi.axhline(
        30,
        linestyle="--"
    )

    ax_rsi.set_ylim(
        0,
        100
    )

    ax_rsi.grid(
        alpha=0.2
    )

    plt.tight_layout()

    st.pyplot(
        fig_rsi,
        use_container_width=True
    )

    plt.close(fig_rsi)


with technical_right:

    st.markdown(
        '<div class="section-title">📉 MACD</div>',
        unsafe_allow_html=True
    )

    fig_macd, ax_macd = plt.subplots(
        figsize=(8, 3)
    )

    ax_macd.plot(
        data.index,
        data["MACD"],
        label="MACD"
    )

    ax_macd.plot(
        data.index,
        data["Signal"],
        label="Signal"
    )

    ax_macd.axhline(
        0,
        linestyle="--"
    )

    ax_macd.grid(
        alpha=0.2
    )

    ax_macd.legend()

    plt.tight_layout()

    st.pyplot(
        fig_macd,
        use_container_width=True
    )

    plt.close(fig_macd)


# ============================================================
# BOLLINGER BANDS
# ============================================================

st.markdown(
    '<div class="section-title">📌 Bollinger Bands</div>',
    unsafe_allow_html=True
)

fig_bb, ax_bb = plt.subplots(
    figsize=(14, 4)
)

ax_bb.plot(
    data.index,
    data["Close"],
    label="Price"
)

ax_bb.plot(
    data.index,
    data["BB_Middle"],
    label="Middle"
)

ax_bb.plot(
    data.index,
    data["BB_Upper"],
    linestyle="--",
    label="Upper"
)

ax_bb.plot(
    data.index,
    data["BB_Lower"],
    linestyle="--",
    label="Lower"
)

ax_bb.fill_between(
    data.index,
    data["BB_Lower"].fillna(0),
    data["BB_Upper"].fillna(0),
    alpha=0.08
)

ax_bb.grid(
    alpha=0.2
)

ax_bb.legend()

plt.tight_layout()

st.pyplot(
    fig_bb,
    use_container_width=True
)

plt.close(fig_bb)


# ============================================================
# NEWS
# ============================================================

st.markdown(
    '<div class="section-title">📰 Latest News Trend</div>',
    unsafe_allow_html=True
)

news_col1, news_col2 = st.columns(
    [1, 3]
)


with news_col1:

    if news_label == "Positive":

        sentiment_icon = "🟢"

    elif news_label == "Negative":

        sentiment_icon = "🔴"

    else:

        sentiment_icon = "🟡"

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                News Sentiment
            </div>

            <div class="metric-value">
                {sentiment_icon} {news_label}
            </div>

            <div class="metric-sub">
                News score: {result["news_score"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with news_col2:

    if news:

        for article in news:

            title = article["title"]
            publisher = article["publisher"]
            link = article["link"]

            if link:

                st.markdown(
                    f"**[{title}]({link})**"
                )

            else:

                st.markdown(
                    f"**{title}**"
                )

            if publisher:

                st.caption(
                    publisher
                )

            st.markdown("---")

    else:

        st.info(
            "No recent news articles were available from the data provider."
        )


# ============================================================
# KEY LEVELS
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Key Technical Levels</div>',
    unsafe_allow_html=True
)

support_20 = float(
    data["Low"].tail(20).min()
)

resistance_20 = float(
    data["High"].tail(20).max()
)

support_50 = float(
    data["Low"].tail(50).min()
)

resistance_50 = float(
    data["High"].tail(50).max()
)

l1, l2, l3, l4 = st.columns(4)

with l1:

    st.metric(
        "20D Support",
        format_currency(support_20)
    )

with l2:

    st.metric(
        "20D Resistance",
        format_currency(resistance_20)
    )

with l3:

    st.metric(
        "50D Support",
        format_currency(support_50)
    )

with l4:

    st.metric(
        "50D Resistance",
        format_currency(resistance_50
    )


# ============================================================
# EXPORT
# ============================================================

st.markdown(
    '<div class="section-title">📁 Export</div>',
    unsafe_allow_html=True
)

export_data = data.copy()

export_data.index = export_data.index.astype(
    str
)

csv = export_data.to_csv().encode(
    "utf-8"
)

st.download_button(
    "⬇️ Download Analysis Data",
    data=csv,
    file_name=f"{symbol}_{market}_analysis.csv",
    mime="text/csv"
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("""
<div class="footer">

    <b>Indian Stock AI Analyzer</b><br><br>

    This application provides technical and news-based analysis
    for informational and research purposes only.

    Buy, Sell and Hold signals are algorithmic estimates and
    should not be considered personalized investment advice.

    Market data may be delayed and predictions can be wrong.

</div>
""", unsafe_allow_html=True)
