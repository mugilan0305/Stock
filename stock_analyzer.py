import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian Stock AI Analyzer",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

NIFTY_10 = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "BHARTIARTL",
    "INFY",
    "TCS",
    "ITC",
    "LT",
    "SBIN",
    "AXISBANK"
]

NIFTY_50 = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "MAXHEALTH",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_yfinance_data(data):
    """
    Normalize Yahoo Finance output.

    New versions of yfinance can return MultiIndex columns even
    for a single ticker. This function converts them into normal
    OHLCV columns.
    """

    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):

        # Try to remove ticker level
        try:
            data.columns = data.columns.get_level_values(0)
        except Exception:
            data.columns = [
                str(col[0]) if isinstance(col, tuple) else str(col)
                for col in data.columns
            ]

    # Make sure columns are strings
    data.columns = [str(c) for c in data.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]

    for col in required:
        if col not in data.columns:
            # Sometimes columns have spaces
            matching = [
                c for c in data.columns
                if str(c).strip().lower() == col.lower()
            ]

            if matching:
                data[col] = data[matching[0]]

    # Convert numeric fields
    for col in required:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    data = data.dropna(subset=["Close"])

    return data


def get_ticker(symbol, market):
    symbol = symbol.upper().strip()

    if market == "NSE":
        return f"{symbol}.NS"

    return f"{symbol}.BO"


def download_stock(symbol, market, start_date, end_date):

    ticker_symbol = get_ticker(symbol, market)

    try:

        # auto_adjust=False is important here.
        # It prevents confusion between adjusted and actual
        # displayed market prices.
        data = yf.download(
            ticker_symbol,
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=False,
            progress=False,
            threads=False
        )

        data = clean_yfinance_data(data)

        if not data.empty:
            return data, ticker_symbol

    except Exception as e:
        st.error(f"Unable to download {ticker_symbol}: {e}")

    return pd.DataFrame(), ticker_symbol


def get_current_price(data):

    if data.empty:
        return None

    try:
        price = float(data["Close"].iloc[-1])

        if np.isfinite(price):
            return price

    except Exception:
        pass

    return None


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(data):

    data = data.copy()

    # Moving averages
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (100 / (1 + rs))

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_Signal"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["MACD_Hist"] = (
        data["MACD"] -
        data["MACD_Signal"]
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    data["BB_Middle"] = data["Close"].rolling(20).mean()

    bb_std = data["Close"].rolling(20).std()

    data["BB_Upper"] = (
        data["BB_Middle"] +
        2 * bb_std
    )

    data["BB_Lower"] = (
        data["BB_Middle"] -
        2 * bb_std
    )

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    data["Return_1D"] = data["Close"].pct_change(1)
    data["Return_5D"] = data["Close"].pct_change(5)
    data["Return_20D"] = data["Close"].pct_change(20)

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    data["Volatility"] = (
        data["Return_1D"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    # --------------------------------------------------------
    # Volume average
    # --------------------------------------------------------

    data["Volume_MA20"] = data["Volume"].rolling(20).mean()

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = data["Close"].shift(1)

    tr1 = data["High"] - data["Low"]

    tr2 = (
        data["High"] -
        previous_close
    ).abs()

    tr3 = (
        data["Low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    data["ATR"] = true_range.rolling(14).mean()

    return data


# ============================================================
# NEWS
# ============================================================

POSITIVE_WORDS = [
    "profit",
    "growth",
    "surge",
    "rises",
    "rise",
    "strong",
    "record",
    "upgrade",
    "bullish",
    "buy",
    "positive",
    "beats",
    "beat",
    "revenue growth",
    "order win",
    "orders",
    "expansion"
]

NEGATIVE_WORDS = [
    "loss",
    "fall",
    "falls",
    "decline",
    "weak",
    "downgrade",
    "bearish",
    "sell",
    "negative",
    "debt",
    "fraud",
    "investigation",
    "penalty",
    "lawsuit",
    "misses",
    "miss",
    "warning"
]


def get_news(symbol):

    articles = []

    try:

        query = requests.utils.quote(
            f"{symbol} India stock"
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
            }
        )

        if response.status_code != 200:
            return articles

        root = ET.fromstring(response.content)

        for item in root.findall(".//item")[:10]:

            title = item.findtext("title")

            link = item.findtext("link")

            pub_date = item.findtext("pubDate")

            if title:

                articles.append({
                    "title": title,
                    "link": link,
                    "date": pub_date
                })

    except Exception:
        return []

    return articles


def calculate_news_sentiment(articles):

    if not articles:
        return 0

    score = 0

    for article in articles:

        title = article["title"].lower()

        for word in POSITIVE_WORDS:
            if word in title:
                score += 1

        for word in NEGATIVE_WORDS:
            if word in title:
                score -= 1

    return score


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(data, news_score=0):

    if data.empty:
        return "NO DATA", 0

    latest = data.iloc[-1]

    price = latest["Close"]

    score = 0

    reasons = []

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    if pd.notna(latest["MA20"]):

        if price > latest["MA20"]:
            score += 1
            reasons.append("Price above MA20")
        else:
            score -= 1
            reasons.append("Price below MA20")

    if pd.notna(latest["MA50"]):

        if price > latest["MA50"]:
            score += 2
            reasons.append("Price above MA50")
        else:
            score -= 2
            reasons.append("Price below MA50")

    if pd.notna(latest["MA200"]):

        if price > latest["MA200"]:
            score += 2
            reasons.append("Long-term trend positive")
        else:
            score -= 2
            reasons.append("Long-term trend negative")

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = latest["RSI"]

    if pd.notna(rsi):

        if 50 <= rsi <= 70:
            score += 1
            reasons.append("RSI supports momentum")

        elif rsi < 30:
            score += 2
            reasons.append("RSI indicates oversold")

        elif rsi > 75:
            score -= 2
            reasons.append("RSI indicates overbought")

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        pd.notna(latest["MACD"]) and
        pd.notna(latest["MACD_Signal"])
    ):

        if latest["MACD"] > latest["MACD_Signal"]:
            score += 2
            reasons.append("MACD bullish")

        else:
            score -= 2
            reasons.append("MACD bearish")

    # --------------------------------------------------------
    # Bollinger
    # --------------------------------------------------------

    if pd.notna(latest["BB_Lower"]):

        if price < latest["BB_Lower"]:
            score += 1
            reasons.append("Price near lower Bollinger Band")

        elif price > latest["BB_Upper"]:
            score -= 1
            reasons.append("Price near upper Bollinger Band")

    # --------------------------------------------------------
    # News
    # --------------------------------------------------------

    if news_score > 2:
        score += 2
        reasons.append("Positive news sentiment")

    elif news_score < -2:
        score -= 2
        reasons.append("Negative news sentiment")

    # --------------------------------------------------------
    # Final signal
    # --------------------------------------------------------

    if score >= 5:
        signal = "STRONG BUY"

    elif score >= 2:
        signal = "BUY"

    elif score <= -5:
        signal = "STRONG SELL"

    elif score <= -2:
        signal = "SELL"

    else:
        signal = "HOLD"

    strength = min(
        100,
        max(
            0,
            50 + score * 8
        )
    )

    return signal, strength, reasons


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def calculate_levels(data):

    recent = data.tail(60)

    support = float(
        recent["Low"].min()
    )

    resistance = float(
        recent["High"].max()
    )

    return support, resistance


# ============================================================
# ML FEATURE ENGINEERING
# ============================================================

FEATURES = [
    "Close",
    "Volume",
    "MA20",
    "MA50",
    "MA200",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "Return_1D",
    "Return_5D",
    "Return_20D",
    "Volatility",
    "ATR"
]


def prepare_ml_data(data, horizon):

    df = data.copy()

    df["Target"] = (
        df["Close"].shift(-horizon)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURES + ["Target"]
    )

    return df


def train_prediction_model(data, horizon):

    df = prepare_ml_data(
        data,
        horizon
    )

    if len(df) < 150:
        return None, None, None

    X = df[FEATURES]
    y = df["Target"]

    split = int(
        len(df) * 0.80
    )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = float(
        model.predict(
            data[FEATURES].iloc[[-1]]
        )[0]
    )

    test_prediction = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    return model, prediction, mae


# ============================================================
# PREDICTION TIMELINE
# ============================================================

def trading_date_from_today(days):

    today = pd.Timestamp.today().normalize()

    business_days = pd.bdate_range(
        start=today + pd.Timedelta(days=1),
        periods=days
    )

    return business_days[-1].date()


def generate_predictions(data):

    horizons = {
        "1 Trading Day": 1,
        "3 Trading Days": 3,
        "5 Trading Days": 5,
        "10 Trading Days": 10,
        "20 Trading Days": 20
    }

    rows = []

    current_price = get_current_price(
        data
    )

    for label, horizon in horizons.items():

        try:

            model, prediction, mae = (
                train_prediction_model(
                    data,
                    horizon
                )
            )

            if prediction is None:
                continue

            expected_return = (
                (prediction / current_price) - 1
            ) * 100

            if expected_return >= 5:
                direction = "Bullish"

            elif expected_return >= 1:
                direction = "Moderately Bullish"

            elif expected_return <= -5:
                direction = "Bearish"

            elif expected_return <= -1:
                direction = "Moderately Bearish"

            else:
                direction = "Neutral"

            rows.append({
                "Timeline": label,
                "Approx. Date": trading_date_from_today(
                    horizon
                ),
                "Current Price": round(
                    current_price,
                    2
                ),
                "Predicted Price": round(
                    prediction,
                    2
                ),
                "Expected Change %": round(
                    expected_return,
                    2
                ),
                "Direction": direction,
                "Model MAE": round(
                    mae,
                    2
                )
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


# ============================================================
# RISK MANAGEMENT
# ============================================================

def calculate_trade_levels(data, signal):

    latest = data.iloc[-1]

    price = float(
        latest["Close"]
    )

    atr = latest["ATR"]

    if pd.isna(atr) or atr <= 0:
        atr = price * 0.02

    if "BUY" in signal:

        stop_loss = price - (
            1.5 * atr
        )

        target_1 = price + (
            2 * atr
        )

        target_2 = price + (
            3 * atr
        )

    elif "SELL" in signal:

        stop_loss = price + (
            1.5 * atr
        )

        target_1 = price - (
            2 * atr
        )

        target_2 = price - (
            3 * atr
        )

    else:

        stop_loss = price - (
            1.5 * atr
        )

        target_1 = price + (
            2 * atr
        )

        target_2 = price + (
            3 * atr
        )

    return (
        round(stop_loss, 2),
        round(target_1, 2),
        round(target_2, 2)
    )


# ============================================================
# SCANNER
# ============================================================

def scan_stocks(symbols, market):

    results = []

    progress = st.progress(0)

    total = len(symbols)

    for i, symbol in enumerate(symbols):

        try:

            end_date = datetime.today().date()

            start_date = (
                end_date -
                timedelta(days=400)
            )

            data, ticker = download_stock(
                symbol,
                market,
                start_date,
                end_date
            )

            if data.empty:
                continue

            data = calculate_indicators(
                data
            )

            news = get_news(symbol)

            news_score = calculate_news_sentiment(
                news
            )

            signal, strength, reasons = (
                generate_signal(
                    data,
                    news_score
                )
            )

            price = get_current_price(
                data
            )

            rsi = data["RSI"].iloc[-1]

            ma20 = data["MA20"].iloc[-1]

            ma50 = data["MA50"].iloc[-1]

            results.append({
                "Stock": symbol,
                "Price": round(price, 2),
                "Signal": signal,
                "Strength": round(strength),
                "RSI": round(rsi, 1)
                if pd.notna(rsi)
                else np.nan,
                "MA20": round(ma20, 2)
                if pd.notna(ma20)
                else np.nan,
                "MA50": round(ma50, 2)
                if pd.notna(ma50)
                else np.nan,
                "News Score": news_score
            })

        except Exception:
            pass

        progress.progress(
            (i + 1) / total
        )

    progress.empty()

    return pd.DataFrame(results)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 Indian Stock AI Analyzer"
)

st.caption(
    "Technical analysis + machine learning + "
    "news sentiment + prediction timeline"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    market = st.selectbox(
        "Market",
        ["NSE", "BSE"]
    )

    analysis_mode = st.selectbox(
        "Analysis Mode",
        [
            "Single Stock",
            "Nifty 10 Scanner",
            "Nifty 50 Scanner"
        ]
    )

    start_date = st.date_input(
        "Historical Start Date",
        datetime.today().date()
        - timedelta(days=5 * 365)
    )

    end_date = st.date_input(
        "Historical End Date",
        datetime.today().date()
    )


# ============================================================
# SCANNER MODE
# ============================================================

if analysis_mode != "Single Stock":

    st.header("🔎 Market Scanner")

    if analysis_mode == "Nifty 10 Scanner":
        stocks = NIFTY_10
        title = "Nifty 10"

    else:
        stocks = NIFTY_50
        title = "Nifty 50"

    st.write(
        f"Scanning **{len(stocks)} stocks** from {title}..."
    )

    if st.button(
        "🚀 Run Market Scanner",
        type="primary"
    ):

        results = scan_stocks(
            stocks,
            market
        )

        if results.empty:

            st.error(
                "No stock data could be retrieved."
            )

        else:

            st.subheader(
                "📊 Market Signals"
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            st.subheader(
                "🟢 Buy Candidates"
            )

            buys = results[
                results["Signal"].str.contains(
                    "BUY",
                    na=False
                )
            ].sort_values(
                "Strength",
                ascending=False
            )

            if buys.empty:
                st.info(
                    "No strong buy candidates found."
                )
            else:
                st.dataframe(
                    buys,
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader(
                "🔴 Sell Candidates"
            )

            sells = results[
                results["Signal"].str.contains(
                    "SELL",
                    na=False
                )
            ].sort_values(
                "Strength",
                ascending=True
            )

            if sells.empty:
                st.info(
                    "No strong sell candidates found."
                )
            else:
                st.dataframe(
                    sells,
                    use_container_width=True,
                    hide_index=True
                )

            csv = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "📥 Download Scanner CSV",
                csv,
                "indian_stock_scanner.csv",
                "text/csv"
            )


# ============================================================
# SINGLE STOCK MODE
# ============================================================

else:

    symbol = st.text_input(
        "Enter Stock Symbol",
        value="TCS",
        placeholder="Example: TCS, RELIANCE, INFY"
    ).upper().strip()

    if symbol:

        st.info(
            f"Fetching {symbol} "
            f"({market})..."
        )

        data, ticker = download_stock(
            symbol,
            market,
            start_date,
            end_date
        )

        if data.empty:

            st.error(
                f"No data found for {ticker}."
            )

            st.stop()

        # ----------------------------------------------------
        # INDICATORS
        # ----------------------------------------------------

        data = calculate_indicators(
            data
        )

        # ----------------------------------------------------
        # NEWS
        # ----------------------------------------------------

        news = get_news(
            symbol
        )

        news_score = calculate_news_sentiment(
            news
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        signal, strength, reasons = (
            generate_signal(
                data,
                news_score
            )
        )

        current_price = get_current_price(
            data
        )

        support, resistance = (
            calculate_levels(data)
        )

        stop_loss, target_1, target_2 = (
            calculate_trade_levels(
                data,
                signal
            )
        )

        latest = data.iloc[-1]

        # ----------------------------------------------------
        # PRICE HEADER
        # ----------------------------------------------------

        st.header(
            f"📊 {symbol} — {market}"
        )

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        col1.metric(
            "Current Price",
            f"₹{current_price:,.2f}"
        )

        col2.metric(
            "Signal",
            signal
        )

        col3.metric(
            "Signal Strength",
            f"{strength}%"
        )

        col4.metric(
            "RSI",
            f"{latest['RSI']:.1f}"
            if pd.notna(latest["RSI"])
            else "N/A"
        )

        col5.metric(
            "News Score",
            news_score
        )

        # ----------------------------------------------------
        # TRADE PLAN
        # ----------------------------------------------------

        st.subheader(
            "🎯 Trade Levels"
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "Support",
            f"₹{support:,.2f}"
        )

        c2.metric(
            "Resistance",
            f"₹{resistance:,.2f}"
        )

        c3.metric(
            "Stop Loss",
            f"₹{stop_loss:,.2f}"
        )

        c4.metric(
            "Target 1",
            f"₹{target_1:,.2f}"
        )

        st.info(
            f"Target 2: ₹{target_2:,.2f}"
        )

        # ----------------------------------------------------
        # SIGNAL REASONS
        # ----------------------------------------------------

        st.subheader(
            "🧠 Why this signal?"
        )

        for reason in reasons:
            st.write(
                f"• {reason}"
            )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        st.subheader(
            "🤖 AI Prediction Timeline"
        )

        with st.spinner(
            "Training prediction models..."
        ):

            predictions = generate_predictions(
                data
            )

        if predictions.empty:

            st.warning(
                "Not enough historical data "
                "to generate predictions."
            )

        else:

            st.dataframe(
                predictions,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Prediction dates are approximate trading dates. "
                "Model predictions are estimates, not guarantees."
            )

        # ----------------------------------------------------
        # PRICE CHART
        # ----------------------------------------------------

        st.subheader(
            "📈 Price & Moving Averages"
        )

        fig1, ax1 = plt.subplots(
            figsize=(14, 6)
        )

        ax1.plot(
            data.index,
            data["Close"],
            label="Close"
        )

        ax1.plot(
            data.index,
            data["MA20"],
            label="MA20"
        )

        ax1.plot(
            data.index,
            data["MA50"],
            label="MA50"
        )

        ax1.plot(
            data.index,
            data["MA200"],
            label="MA200"
        )

        ax1.set_title(
            f"{symbol} Price Trend"
        )

        ax1.legend()
        ax1.grid(True)

        st.pyplot(fig1)

        plt.close(fig1)

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        st.subheader(
            "🌀 RSI"
        )

        fig2, ax2 = plt.subplots(
            figsize=(14, 4)
        )

        ax2.plot(
            data.index,
            data["RSI"],
            label="RSI"
        )

        ax2.axhline(
            70,
            linestyle="--"
        )

        ax2.axhline(
            30,
            linestyle="--"
        )

        ax2.set_title(
            "Relative Strength Index"
        )

        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig2)

        plt.close(fig2)

        # ----------------------------------------------------
        # MACD
        # ----------------------------------------------------

        st.subheader(
            "📉 MACD"
        )

        fig3, ax3 = plt.subplots(
            figsize=(14, 4)
        )

        ax3.plot(
            data.index,
            data["MACD"],
            label="MACD"
        )

        ax3.plot(
            data.index,
            data["MACD_Signal"],
            label="Signal"
        )

        ax3.bar(
            data.index,
            data["MACD_Hist"],
            alpha=0.3,
            label="Histogram"
        )

        ax3.legend()
        ax3.grid(True)

        st.pyplot(fig3)

        plt.close(fig3)

        # ----------------------------------------------------
        # BOLLINGER BANDS
        # ----------------------------------------------------

        st.subheader(
            "📌 Bollinger Bands"
        )

        fig4, ax4 = plt.subplots(
            figsize=(14, 5)
        )

        ax4.plot(
            data.index,
            data["Close"],
            label="Close"
        )

        ax4.plot(
            data.index,
            data["BB_Upper"],
            linestyle="--",
            label="Upper"
        )

        ax4.plot(
            data.index,
            data["BB_Middle"],
            label="Middle"
        )

        ax4.plot(
            data.index,
            data["BB_Lower"],
            linestyle="--",
            label="Lower"
        )

        ax4.legend()
        ax4.grid(True)

        st.pyplot(fig4)

        plt.close(fig4)

        # ----------------------------------------------------
        # NEWS
        # ----------------------------------------------------

        st.subheader(
            "📰 Latest News"
        )

        if not news:

            st.info(
                "No recent news was retrieved."
            )

        else:

            for article in news[:10]:

                title = article["title"]

                link = article["link"]

                if link:

                    st.markdown(
                        f"• [{title}]({link})"
                    )

                else:

                    st.write(
                        f"• {title}"
                    )

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        st.subheader(
            "📋 Latest Market Data"
        )

        display_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "MA20",
            "MA50",
            "MA200",
            "RSI",
            "MACD",
            "MACD_Signal",
            "ATR"
        ]

        available_columns = [
            col
            for col in display_columns
            if col in data.columns
        ]

        st.dataframe(
            data[
                available_columns
            ].tail(20),
            use_container_width=True
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        st.subheader(
            "📁 Export Analysis"
        )

        csv = data.to_csv(
            index=True
        ).encode("utf-8")

        st.download_button(
            "📥 Download Complete CSV",
            csv,
            f"{symbol}_{market}_analysis.csv",
            "text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚠️ Educational analysis only. "
    "AI predictions, technical indicators and news sentiment "
    "can be wrong and should not be treated as guaranteed "
    "investment advice."
)
