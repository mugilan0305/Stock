import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor


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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: #0b1120;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .dashboard-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        font-size: 17px;
        color: #94a3b8;
        margin-bottom: 20px;
    }

    .status-box {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .signal-buy {
        background: #052e16;
        border: 1px solid #16a34a;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .signal-sell {
        background: #450a0a;
        border: 1px solid #dc2626;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .signal-hold {
        background: #422006;
        border: 1px solid #f59e0b;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .signal-text {
        font-size: 30px;
        font-weight: 800;
    }

    .small-text {
        color: #94a3b8;
        font-size: 13px;
    }

    .prediction-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }

    .prediction-price {
        font-size: 25px;
        font-weight: 700;
        color: #f8fafc;
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

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="dashboard-title">📈 Indian Stock AI Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Live market price • Technical analysis • AI predictions • Trading signals • News sentiment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# STOCK UNIVERSE
# ============================================================

NIFTY_50 = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "INFY": "INFY.NS",
    "ITC": "ITC.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "SBIN": "SBIN.NS",
    "LT": "LT.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "MARUTI": "MARUTI.NS",
    "M&M": "M&M.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "NTPC": "NTPC.NS",
    "POWERGRID": "POWERGRID.NS",
    "HCLTECH": "HCLTECH.NS",
    "WIPRO": "WIPRO.NS",
    "TECHM": "TECHM.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "COALINDIA": "COALINDIA.NS",
    "ONGC": "ONGC.NS",
    "BEL": "BEL.NS",
    "BPCL": "BPCL.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "GRASIM": "GRASIM.NS",
    "HINDALCO": "HINDALCO.NS",
    "CIPLA": "CIPLA.NS",
    "DRREDDY": "DRREDDY.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "TRENT": "TRENT.NS",
    "INDUSINDBK": "INDUSINDBK.NS"
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Analysis Settings")

market = st.sidebar.selectbox(
    "Market",
    ["NSE", "BSE"]
)

symbol = st.sidebar.text_input(
    "Stock Symbol",
    value="TCS"
).strip().upper()

analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    [
        "Single Stock",
        "NIFTY 50 Scanner"
    ]
)

run_analysis = st.sidebar.button(
    "🔍 Analyze Stock",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Prediction horizons**

    • 1 Trading Day  
    • 5 Trading Days  
    • 20 Trading Days  

    Predictions are estimates based on technical patterns and should not be treated as guaranteed prices.
    """
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_ticker(symbol, market):
    if market == "NSE":
        return f"{symbol}.NS"
    return f"{symbol}.BO"


def flatten_columns(df):
    """
    Handles newer yfinance versions where downloaded data
    can sometimes contain MultiIndex columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    return df


def get_market_data(ticker_symbol):
    """
    Downloads approximately one year of data internally.
    Historical data is used for analysis but is NOT displayed
    as a historical table.
    """

    try:
        ticker = yf.Ticker(ticker_symbol)

        data = ticker.history(
            period="1y",
            interval="1d",
            auto_adjust=False
        )

        if data is None or data.empty:
            return None, None

        data = flatten_columns(data)

        data = data.dropna(
            subset=["Close"]
        )

        if data.empty:
            return None, None

        # ----------------------------------------------------
        # Get latest price
        # ----------------------------------------------------

        current_price = None

        try:
            fast_info = ticker.fast_info

            if fast_info:
                current_price = fast_info.get(
                    "last_price"
                )
        except Exception:
            current_price = None

        # Fallback to latest historical close
        if current_price is None or pd.isna(current_price):
            current_price = float(
                data["Close"].iloc[-1]
            )

        current_price = float(current_price)

        return data, current_price

    except Exception as e:
        return None, None


def calculate_indicators(data):

    df = data.copy()

    close = df["Close"]

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["BB_MIDDLE"] = bb_middle
    df["BB_UPPER"] = bb_middle + (
        2 * bb_std
    )
    df["BB_LOWER"] = bb_middle - (
        2 * bb_std
    )

    # --------------------------------------------------------
    # Daily return
    # --------------------------------------------------------

    df["RETURN"] = close.pct_change()

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["VOLATILITY"] = (
        df["RETURN"].rolling(20).std()
        * np.sqrt(252)
        * 100
    )

    # Average True Range (ATR) for dynamic stop-loss levels
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    return df


# ============================================================
# TECHNICAL SIGNAL
# ============================================================

def calculate_signal(df):

    latest = df.iloc[-1]

    score = 0
    reasons = []

    # --------------------------------------------------------
    # Price vs MA20
    # --------------------------------------------------------

    if pd.notna(latest["MA20"]):

        if latest["Close"] > latest["MA20"]:
            score += 1
            reasons.append(
                "Price is above MA20"
            )
        else:
            score -= 1
            reasons.append(
                "Price is below MA20"
            )

    # --------------------------------------------------------
    # MA20 vs MA50
    # --------------------------------------------------------

    if pd.notna(latest["MA50"]):

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

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = latest["RSI"]

    if pd.notna(rsi):

        if rsi < 30:
            score += 2
            reasons.append(
                "RSI indicates oversold conditions"
            )

        elif rsi > 70:
            score -= 2
            reasons.append(
                "RSI indicates overbought conditions"
            )

        elif rsi >= 50:
            score += 1
            reasons.append(
                "RSI has bullish momentum"
            )

        else:
            score -= 1
            reasons.append(
                "RSI has weak momentum"
            )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        pd.notna(latest["MACD"])
        and pd.notna(latest["MACD_SIGNAL"])
    ):

        if latest["MACD"] > latest["MACD_SIGNAL"]:
            score += 2
            reasons.append(
                "MACD is bullish"
            )
        else:
            score -= 2
            reasons.append(
                "MACD is bearish"
            )

    # --------------------------------------------------------
    # Final signal
    # --------------------------------------------------------

    if score >= 3:
        signal = "BUY"

    elif score <= -3:
        signal = "SELL"

    else:
        signal = "HOLD"

    max_score = 8

    strength = min(
        100,
        max(
            0,
            int(
                abs(score)
                / max_score
                * 100
            )
        )
    )

    return signal, score, strength, reasons


# ============================================================
# AI PREDICTION
# ============================================================

def create_prediction(df):

    work = df.copy()

    work["Return_1"] = work["Close"].pct_change()

    work["Return_5"] = work["Close"].pct_change(5)

    work["MA20_DIFF"] = (
        work["Close"] / work["MA20"] - 1
    )

    work["MA50_DIFF"] = (
        work["Close"] / work["MA50"] - 1
    )

    work["RSI_FEATURE"] = work["RSI"]

    work["MACD_DIFF"] = (
        work["MACD"]
        - work["MACD_SIGNAL"]
    )

    work["Volatility"] = (
        work["RETURN"]
        .rolling(20)
        .std()
    )

    feature_columns = [
        "Return_1",
        "Return_5",
        "MA20_DIFF",
        "MA50_DIFF",
        "RSI_FEATURE",
        "MACD_DIFF",
        "Volatility"
    ]

    work = work.dropna(
        subset=feature_columns + ["Close"]
    )

    if len(work) < 80:
        return None

    X = work[feature_columns]
    y = work["Close"]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        random_state=42,
        min_samples_leaf=3
    )

    model.fit(X, y)

    latest_features = work[
        feature_columns
    ].iloc[-1].values.reshape(1, -1)

    current_price = float(
        df["Close"].iloc[-1]
    )

    predictions = {}

    horizons = {
        "1 Trading Day": 1,
        "5 Trading Days": 5,
        "20 Trading Days": 20
    }

    # Base model estimate
    base_prediction = float(
        model.predict(
            latest_features
        )[0]
    )

    current = current_price

    for label, days in horizons.items():

        # Scale the model's expected move
        raw_change = (
            base_prediction - current
        )

        horizon_factor = np.sqrt(days)

        projected_change = (
            raw_change
            * horizon_factor
            * 0.35
        )

        predicted_price = (
            current
            + projected_change
        )

        # Prevent unrealistic model jumps
        max_move = current * (
            0.04 * np.sqrt(days)
        )

        lower_bound = current - max_move
        upper_bound = current + max_move

        predicted_price = max(
            lower_bound,
            min(
                upper_bound,
                predicted_price
            )
        )

        change_pct = (
            predicted_price
            / current
            - 1
        ) * 100

        predictions[label] = {
            "days": days,
            "price": float(
                predicted_price
            ),
            "change_pct": float(
                change_pct
            )
        }

    return predictions


# ============================================================
# NEWS SENTIMENT
# ============================================================

def get_news_sentiment(ticker_symbol):

    positive_words = [
        "growth",
        "profit",
        "surge",
        "rises",
        "rise",
        "strong",
        "positive",
        "upgrade",
        "buy",
        "record",
        "beat",
        "bullish",
        "expansion"
    ]

    negative_words = [
        "fall",
        "falls",
        "drop",
        "loss",
        "weak",
        "negative",
        "downgrade",
        "sell",
        "bearish",
        "decline",
        "concern",
        "risk",
        "cut"
    ]

    try:

        ticker = yf.Ticker(
            ticker_symbol
        )

        news = ticker.news

        if not news:
            return {
                "label": "No recent news",
                "score": 0,
                "headlines": []
            }

        headlines = []

        score = 0

        for item in news[:10]:

            title = item.get(
                "title",
                ""
            )

            if not title:
                continue

            headlines.append(title)

            text = title.lower()

            positive_count = sum(
                word in text
                for word in positive_words
            )

            negative_count = sum(
                word in text
                for word in negative_words
            )

            score += (
                positive_count
                - negative_count
            )

        if score >= 2:
            label = "Positive"

        elif score <= -2:
            label = "Negative"

        else:
            label = "Neutral"

        return {
            "label": label,
            "score": score,
            "headlines": headlines
        }

    except Exception:

        return {
            "label": "News unavailable",
            "score": 0,
            "headlines": []
        }


# ============================================================
# PREDICTION REASON + TRADE LEVELS
# ============================================================

def build_prediction_reason(df, prediction, news_label):
    latest = df.iloc[-1]
    reasons = []

    # Trend
    if pd.notna(latest["MA20"]):
        if latest["Close"] > latest["MA20"]:
            reasons.append("price is above the 20-day moving average")
        else:
            reasons.append("price is below the 20-day moving average")

    if pd.notna(latest["MA50"]):
        if latest["MA20"] > latest["MA50"]:
            reasons.append("the short-term trend is stronger than the 50-day trend")
        else:
            reasons.append("the short-term trend is weaker than the 50-day trend")

    # RSI
    rsi = latest["RSI"]
    if pd.notna(rsi):
        if rsi >= 60:
            reasons.append(f"RSI at {rsi:.1f} shows positive momentum")
        elif rsi <= 40:
            reasons.append(f"RSI at {rsi:.1f} shows weak momentum")
        else:
            reasons.append(f"RSI at {rsi:.1f} is in a relatively neutral zone")

    # MACD
    if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_SIGNAL"]):
        if latest["MACD"] > latest["MACD_SIGNAL"]:
            reasons.append("MACD is above its signal line, supporting upside momentum")
        else:
            reasons.append("MACD is below its signal line, limiting upside momentum")

    # News
    if news_label == "Positive":
        reasons.append("recent news sentiment is positive")
    elif news_label == "Negative":
        reasons.append("recent news sentiment is negative")
    elif news_label == "Neutral":
        reasons.append("recent news sentiment is mixed or neutral")

    # Model direction
    change = prediction["change_pct"]
    if change > 0:
        reasons.append(f"the model projects approximately {change:+.2f}% upside over this horizon")
    elif change < 0:
        reasons.append(f"the model projects approximately {change:+.2f}% downside over this horizon")
    else:
        reasons.append("the model projects limited price movement over this horizon")

    # Keep the explanation readable
    return "; ".join(reasons[:6]) + "."


def calculate_trade_levels(df, current_price, target_price):
    latest = df.iloc[-1]
    atr = latest.get("ATR", np.nan)

    if pd.isna(atr) or atr <= 0:
        atr = current_price * 0.02

    recent_support = df["Low"].rolling(20).min().iloc[-1]
    recent_resistance = df["High"].rolling(20).max().iloc[-1]

    if target_price >= current_price:
        atr_stop = current_price - (1.5 * atr)
        support_stop = recent_support * 0.98 if pd.notna(recent_support) else atr_stop
        stop_loss = max(atr_stop, support_stop)
        stop_loss = min(stop_loss, current_price * 0.99)
    else:
        atr_stop = current_price + (1.5 * atr)
        resistance_stop = recent_resistance * 1.02 if pd.notna(recent_resistance) else atr_stop
        stop_loss = min(atr_stop, resistance_stop)
        stop_loss = max(stop_loss, current_price * 1.01)

    risk = abs(current_price - stop_loss)
    reward = abs(target_price - current_price)
    risk_reward = reward / risk if risk > 0 else 0

    target_pct = ((target_price - current_price) / current_price) * 100
    stop_pct = ((stop_loss - current_price) / current_price) * 100

    return {
        "stop_loss": float(stop_loss),
        "risk_reward": float(risk_reward),
        "target_pct": float(target_pct),
        "stop_pct": float(stop_pct)
    }


def prediction_timeline(predictions):

    if not predictions:
        return []

    today = datetime.now()
    result = []

    for label, item in predictions.items():

        days = item["days"]

        estimated_date = today + timedelta(
            days=days
        )

        result.append({
            "Horizon": label,
            "Estimated Date": estimated_date.strftime(
                "%d %b %Y"
            ),
            "Target Price": item["price"],
            "Expected Change": item["change_pct"],
            "Stop Loss": item.get("stop_loss"),
            "Risk / Reward": item.get("risk_reward"),
            "Reason": item.get("reason", "")
        })

    return result


# ============================================================
# STOCK ANALYSIS
# ============================================================

def analyze_stock(symbol, market):

    ticker_symbol = get_ticker(
        symbol,
        market
    )

    data, current_price = get_market_data(
        ticker_symbol
    )

    if data is None:
        return None

    indicators = calculate_indicators(
        data
    )

    signal, score, strength, reasons = calculate_signal(
        indicators
    )

    predictions = create_prediction(
        indicators
    )

    news = get_news_sentiment(
        ticker_symbol
    )

    # Add a transparent explanation and trade levels to each horizon.
    if predictions:
        for label, prediction in predictions.items():
            trade_levels = calculate_trade_levels(
                indicators,
                current_price,
                prediction["price"]
            )
            prediction.update(trade_levels)
            prediction["reason"] = build_prediction_reason(
                indicators,
                prediction,
                news["label"]
            )

    return {
        "symbol": symbol,
        "ticker": ticker_symbol,
        "data": indicators,
        "price": current_price,
        "signal": signal,
        "score": score,
        "strength": strength,
        "reasons": reasons,
        "predictions": predictions,
        "news": news
    }


# ============================================================
# EMPTY STATE
# ============================================================

if not run_analysis:

    st.markdown(
        """
        <div style="
            background:#111827;
            border:1px solid #263244;
            border-radius:18px;
            padding:60px;
            text-align:center;
            margin-top:30px;
        ">

        <div style="font-size:55px;">📊</div>

        <h2 style="color:#f8fafc;">
        Start Your Analysis
        </h2>

        <p style="color:#94a3b8;font-size:17px;">
        Enter a stock symbol in the sidebar and click
        <b>Analyze Stock</b> to view the latest market
        price, technical indicators, AI prediction and
        BUY / SELL / HOLD signal.
        </p>

        <p style="color:#64748b;">
        Examples: TCS • RELIANCE • INFY • HDFCBANK
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# NIFTY 50 SCANNER
# ============================================================

if analysis_mode == "NIFTY 50 Scanner":

    st.markdown(
        '<div class="section-title">🇮🇳 NIFTY 50 Market Scanner</div>',
        unsafe_allow_html=True
    )

    results = []

    progress = st.progress(0)

    total = len(NIFTY_50)

    for index, (
        stock,
        ticker
    ) in enumerate(
        NIFTY_50.items()
    ):

        try:

            data, price = get_market_data(
                ticker
            )

            if data is None:
                continue

            indicators = calculate_indicators(
                data
            )

            signal, score, strength, reasons = calculate_signal(
                indicators
            )

            results.append({
                "Stock": stock,
                "Price": round(
                    price,
                    2
                ),
                "Signal": signal,
                "Score": score,
                "Strength": f"{strength}%",
                "RSI": round(
                    indicators["RSI"].iloc[-1],
                    1
                ) if pd.notna(
                    indicators["RSI"].iloc[-1]
                ) else np.nan
            })

        except Exception:
            pass

        progress.progress(
            (index + 1) / total
        )

    progress.empty()

    if results:

        scanner_df = pd.DataFrame(
            results
        )

        scanner_df = scanner_df.sort_values(
            by="Score",
            ascending=False
        )

        st.dataframe(
            scanner_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download NIFTY 50 Scanner",
            data=scanner_df.to_csv(
                index=False
            ),
            file_name="nifty50_scanner.csv",
            mime="text/csv"
        )

    else:

        st.error(
            "Unable to retrieve NIFTY 50 data right now."
        )

    st.stop()


# ============================================================
# SINGLE STOCK ANALYSIS
# ============================================================

result = analyze_stock(
    symbol,
    market
)

if result is None:

    st.error(
        f"""
        Unable to retrieve data for **{symbol}**.

        Please check:

        • Stock symbol  
        • NSE/BSE selection  
        • Internet connection  
        • Yahoo Finance availability
        """
    )

    st.stop()


# ============================================================
# CURRENT PRICE
# ============================================================

current_price = result["price"]

signal = result["signal"]

score = result["score"]

strength = result["strength"]

news = result["news"]

updated_time = datetime.now().strftime(
    "%d %b %Y • %I:%M:%S %p"
)


# ============================================================
# MARKET STATUS
# ============================================================

st.markdown(
    f"""
    <div class="status-box">
        🟢 <b>Market data connected</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        {result["ticker"]}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Updated: {updated_time}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PRICE + SIGNAL METRICS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Current Price",
        f"₹{current_price:,.2f}"
    )

with col2:

    st.metric(
        "Signal",
        signal
    )

with col3:

    st.metric(
        "Signal Score",
        f"{score:+d}"
    )

with col4:

    st.metric(
        "Signal Strength",
        f"{strength}%"
    )

with col5:

    st.metric(
        "News Sentiment",
        news["label"]
    )


# ============================================================
# SIGNAL CARD
# ============================================================

if signal == "BUY":

    st.markdown(
        f"""
        <div class="signal-buy">
            <div class="signal-text">
                🟢 BUY
            </div>
            <div>
                Technical strength: {strength}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

elif signal == "SELL":

    st.markdown(
        f"""
        <div class="signal-sell">
            <div class="signal-text">
                🔴 SELL
            </div>
            <div>
                Technical weakness: {strength}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        f"""
        <div class="signal-hold">
            <div class="signal-text">
                🟡 HOLD
            </div>
            <div>
                Current technical score: {score:+d}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTIONS
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Price Prediction</div>',
    unsafe_allow_html=True
)

predictions = result["predictions"]

if predictions:

    p1, p2, p3 = st.columns(3)

    prediction_items = list(predictions.items())

    for column, (label, prediction) in zip(
        [p1, p2, p3],
        prediction_items
    ):

        change = prediction["change_pct"]
        stop_loss = prediction.get("stop_loss")
        risk_reward = prediction.get("risk_reward", 0)
        reason = prediction.get(
            "reason",
            "Prediction is based on the technical and market factors shown below."
        )

        change_class = (
            "positive"
            if change > 0
            else "negative"
            if change < 0
            else "neutral"
        )

        direction = (
            "↑"
            if change > 0
            else "↓"
            if change < 0
            else "→"
        )

        estimated_date = (
            datetime.now()
            + timedelta(days=prediction["days"])
        ).strftime("%d %b %Y")

        target_label = "Target" if change >= 0 else "Downside Target"

        with column:

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="small-text">
                        {label}
                    </div>

                    <div class="prediction-price">
                        ₹{prediction["price"]:,.2f}
                    </div>

                    <div class="{change_class}">
                        {direction} {change:+.2f}%
                    </div>

                    <div style="margin-top:14px; text-align:left;">
                        <div class="small-text">{target_label}</div>
                        <div style="font-size:17px;font-weight:700;color:#f8fafc;">
                            ₹{prediction["price"]:,.2f}
                        </div>

                        <div class="small-text" style="margin-top:10px;">Stop Loss</div>
                        <div style="font-size:17px;font-weight:700;color:#ef4444;">
                            ₹{stop_loss:,.2f}
                        </div>

                        <div class="small-text" style="margin-top:10px;">Risk / Reward</div>
                        <div style="font-size:17px;font-weight:700;color:#60a5fa;">
                            1 : {risk_reward:.2f}
                        </div>

                        <div class="small-text" style="margin-top:12px;">Estimated date</div>
                        <div style="font-size:14px;color:#f8fafc;">
                            {estimated_date}
                        </div>
                    </div>

                    <div style="margin-top:15px;text-align:left;">
                        <div style="color:#f8fafc;font-size:14px;font-weight:700;margin-bottom:5px;">
                            Why this prediction?
                        </div>
                        <div style="color:#94a3b8;font-size:13px;line-height:1.55;">
                            {reason}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# TIMELINE
# ============================================================

st.markdown(
    '<div class="section-title">🗓️ Prediction Timeline</div>',
    unsafe_allow_html=True
)

timeline = prediction_timeline(
    predictions
)

if timeline:

    timeline_df = pd.DataFrame(timeline)

    timeline_df["Target Price"] = timeline_df["Target Price"].map(
        lambda x: f"₹{x:,.2f}"
    )

    timeline_df["Expected Change"] = timeline_df["Expected Change"].map(
        lambda x: f"{x:+.2f}%"
    )

    timeline_df["Stop Loss"] = timeline_df["Stop Loss"].map(
        lambda x: f"₹{x:,.2f}"
    )

    timeline_df["Risk / Reward"] = timeline_df["Risk / Reward"].map(
        lambda x: f"1 : {x:.2f}"
    )

    st.dataframe(
        timeline_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

df = result["data"]

latest = df.iloc[-1]

st.markdown(
    '<div class="section-title">📊 Technical Indicators</div>',
    unsafe_allow_html=True
)

t1, t2, t3, t4 = st.columns(4)

with t1:

    rsi_value = latest["RSI"]

    st.metric(
        "RSI",
        f"{rsi_value:.1f}"
        if pd.notna(rsi_value)
        else "N/A"
    )

with t2:

    macd_value = latest["MACD"]

    st.metric(
        "MACD",
        f"{macd_value:.2f}"
        if pd.notna(macd_value)
        else "N/A"
    )

with t3:

    ma20 = latest["MA20"]

    st.metric(
        "MA20",
        f"₹{ma20:,.2f}"
        if pd.notna(ma20)
        else "N/A"
    )

with t4:

    ma50 = latest["MA50"]

    st.metric(
        "MA50",
        f"₹{ma50:,.2f}"
        if pd.notna(ma50)
        else "N/A"
    )


# ============================================================
# SIGNAL REASONS
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Why This Signal?</div>',
    unsafe_allow_html=True
)

for reason in result["reasons"]:

    st.write(
        f"• {reason}"
    )


# ============================================================
# PRICE CHART
# ============================================================

st.markdown(
    '<div class="section-title">📈 Technical Price View</div>',
    unsafe_allow_html=True
)

chart_df = df.tail(180)

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    chart_df.index,
    chart_df["Close"],
    label="Price"
)

ax.plot(
    chart_df.index,
    chart_df["MA20"],
    label="MA20"
)

ax.plot(
    chart_df.index,
    chart_df["MA50"],
    label="MA50"
)

ax.set_title(
    f"{symbol} - Technical Price Analysis"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Price (₹)"
)

ax.grid(
    True,
    alpha=0.2
)

ax.legend()

fig.tight_layout()

st.pyplot(
    fig,
    clear_figure=True
)


# ============================================================
# RSI CHART
# ============================================================

st.markdown(
    '<div class="section-title">🌀 RSI Momentum</div>',
    unsafe_allow_html=True
)

fig2, ax2 = plt.subplots(
    figsize=(14, 3)
)

ax2.plot(
    chart_df.index,
    chart_df["RSI"],
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

ax2.set_ylim(
    0,
    100
)

ax2.set_title(
    "Relative Strength Index"
)

ax2.grid(
    True,
    alpha=0.2
)

ax2.legend()

fig2.tight_layout()

st.pyplot(
    fig2,
    clear_figure=True
)


# ============================================================
# MACD CHART
# ============================================================

st.markdown(
    '<div class="section-title">📉 MACD</div>',
    unsafe_allow_html=True
)

fig3, ax3 = plt.subplots(
    figsize=(14, 3)
)

ax3.plot(
    chart_df.index,
    chart_df["MACD"],
    label="MACD"
)

ax3.plot(
    chart_df.index,
    chart_df["MACD_SIGNAL"],
    label="Signal"
)

ax3.set_title(
    "MACD & Signal Line"
)

ax3.grid(
    True,
    alpha=0.2
)

ax3.legend()

fig3.tight_layout()

st.pyplot(
    fig3,
    clear_figure=True
)


# ============================================================
# BOLLINGER BANDS
# ============================================================

st.markdown(
    '<div class="section-title">📌 Bollinger Bands</div>',
    unsafe_allow_html=True
)

fig4, ax4 = plt.subplots(
    figsize=(14, 4)
)

ax4.plot(
    chart_df.index,
    chart_df["Close"],
    label="Price"
)

ax4.plot(
    chart_df.index,
    chart_df["BB_MIDDLE"],
    label="Middle"
)

ax4.plot(
    chart_df.index,
    chart_df["BB_UPPER"],
    linestyle="--",
    label="Upper"
)

ax4.plot(
    chart_df.index,
    chart_df["BB_LOWER"],
    linestyle="--",
    label="Lower"
)

ax4.set_title(
    "Bollinger Bands"
)

ax4.grid(
    True,
    alpha=0.2
)

ax4.legend()

fig4.tight_layout()

st.pyplot(
    fig4,
    clear_figure=True
)


# ============================================================
# NEWS
# ============================================================

st.markdown(
    '<div class="section-title">📰 Latest News Sentiment</div>',
    unsafe_allow_html=True
)

st.info(
    f"Overall news sentiment: **{news['label']}**"
)

if news["headlines"]:

    for headline in news["headlines"][:8]:

        st.write(
            f"• {headline}"
        )

else:

    st.write(
        "No recent news headlines were available."
    )


# ============================================================
# EXPORT
# ============================================================

st.markdown(
    '<div class="section-title">📁 Export Analysis</div>',
    unsafe_allow_html=True
)

export_df = pd.DataFrame({
    "Stock": [symbol],
    "Ticker": [result["ticker"]],
    "Current Price": [current_price],
    "Signal": [signal],
    "Signal Score": [score],
    "Signal Strength": [strength],
    "RSI": [latest["RSI"]],
    "MACD": [latest["MACD"]],
    "MA20": [latest["MA20"]],
    "MA50": [latest["MA50"]],
    "News Sentiment": [news["label"]],
    "1D Target": [predictions.get("1 Trading Day", {}).get("price") if predictions else None],
    "5D Target": [predictions.get("5 Trading Days", {}).get("price") if predictions else None],
    "20D Target": [predictions.get("20 Trading Days", {}).get("price") if predictions else None],
    "1D Stop Loss": [predictions.get("1 Trading Day", {}).get("stop_loss") if predictions else None],
    "5D Stop Loss": [predictions.get("5 Trading Days", {}).get("stop_loss") if predictions else None],
    "20D Stop Loss": [predictions.get("20 Trading Days", {}).get("stop_loss") if predictions else None],
    "Updated": [updated_time]
})

csv_data = export_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Download Analysis CSV",
    data=csv_data,
    file_name=f"{symbol}_analysis.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.caption(
    """
    ⚠️ This application provides algorithmic estimates based on
    market data, technical indicators, machine-learning patterns
    and available news headlines. Predictions are not guaranteed
    and should not be considered financial advice.
    """
)
