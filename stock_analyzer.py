import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ==============================
       MAIN PAGE
       ============================== */

    .stApp {
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(37, 99, 235, 0.12),
                transparent 30%
            ),
            #0b1120;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }

    section[data-testid="stSidebar"] h2 {
        color: #f8fafc;
    }

    /* ==============================
       HEADER
       ============================== */

    .dashboard-header {
        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.95),
                rgba(15, 23, 42, 0.98)
            );
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 22px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.22);
    }

    .dashboard-title {
        font-size: 34px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        color: #f8fafc;
    }

    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 6px;
    }

    .market-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 15px;
        padding: 7px 13px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #86efac;
        font-size: 13px;
        font-weight: 600;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px rgba(34, 197, 94, 0.8);
    }

    /* ==============================
       CARDS
       ============================== */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.95),
                rgba(15, 23, 42, 0.98)
            );
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 27px;
        font-weight: 800;
        margin-top: 10px;
    }

    .metric-small {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ==============================
       PRICE CARD
       ============================== */

    .price-card {
        background:
            linear-gradient(
                135deg,
                rgba(30, 64, 175, 0.28),
                rgba(15, 23, 42, 0.98)
            );
        border: 1px solid rgba(59, 130, 246, 0.30);
        border-radius: 22px;
        padding: 28px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.20);
    }

    .price-label {
        color: #93c5fd;
        font-size: 14px;
        font-weight: 600;
    }

    .price-value {
        font-size: 44px;
        font-weight: 850;
        color: #ffffff;
        margin-top: 5px;
        letter-spacing: -1px;
    }

    .timestamp {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 8px;
    }

    /* ==============================
       SIGNAL
       ============================== */

    .signal-buy {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #86efac;
    }

    .signal-sell {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #fca5a5;
    }

    .signal-hold {
        background: rgba(234, 179, 8, 0.12);
        border: 1px solid rgba(234, 179, 8, 0.35);
        color: #fde68a;
    }

    .signal-box {
        border-radius: 22px;
        padding: 25px;
        text-align: center;
        height: 100%;
    }

    .signal-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        opacity: 0.8;
    }

    .signal-value {
        font-size: 38px;
        font-weight: 900;
        margin-top: 8px;
    }

    .signal-score {
        font-size: 14px;
        margin-top: 5px;
        opacity: 0.75;
    }

    /* ==============================
       SECTION TITLES
       ============================== */

    .section-title {
        color: #f8fafc;
        font-size: 21px;
        font-weight: 750;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-top: -10px;
        margin-bottom: 15px;
    }

    /* ==============================
       FOOTER
       ============================== */

    .footer {
        margin-top: 35px;
        padding: 20px;
        text-align: center;
        color: #64748b;
        font-size: 12px;
        border-top: 1px solid rgba(148, 163, 184, 0.10);
    }

    /* ==============================
       STREAMLIT BUTTON
       ============================== */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }

    /* ==============================
       MOBILE
       ============================== */

    @media (max-width: 768px) {

        .dashboard-title {
            font-size: 26px;
        }

        .price-value {
            font-size: 34px;
        }

        .signal-value {
            font-size: 30px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">

        <div class="dashboard-title">
            📈 Indian Stock Analyzer
        </div>

        <div class="dashboard-subtitle">
            Real-time market view • Technical analysis • Trading signals
        </div>

        <div class="market-status">
            <span class="status-dot"></span>
            Market data connected
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Analysis Settings")

    market = st.selectbox(
        "Market",
        ["NSE", "BSE"]
    )

    symbol = st.text_input(
        "Stock Symbol",
        placeholder="TCS",
        help="Enter an NSE/BSE stock symbol"
    ).strip().upper()

    st.markdown("---")

    st.markdown("### 📊 Indicators")

    st.checkbox(
        "Moving Averages",
        value=True,
        disabled=True
    )

    st.checkbox(
        "RSI",
        value=True,
        disabled=True
    )

    st.checkbox(
        "MACD",
        value=True,
        disabled=True
    )

    st.checkbox(
        "Bollinger Bands",
        value=True,
        disabled=True
    )

    st.markdown("---")

    refresh = st.button(
        "🔄 Refresh Market Data",
        use_container_width=True
    )

    st.markdown("---")

    st.caption(
        "Data source: Yahoo Finance"
    )

    st.caption(
        "Technical signals are for informational purposes only."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def flatten_columns(df):

    if df is None or df.empty:
        return df

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    return df


def get_close_series(df):

    df = flatten_columns(df)

    if "Close" not in df.columns:
        return pd.Series(dtype=float)

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(
        close,
        errors="coerce"
    )

    return close.dropna()


def calculate_indicators(data):

    df = flatten_columns(data.copy())

    close = get_close_series(df)

    if close.empty:
        return pd.DataFrame()

    df["Close"] = close

    # Moving averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(20).mean()

    std = df["Close"].rolling(20).std()

    df["BB_Upper"] = (
        df["BB_Middle"] + 2 * std
    )

    df["BB_Lower"] = (
        df["BB_Middle"] - 2 * std
    )

    return df


def generate_signal(df):

    if df.empty or len(df) < 30:

        return {
            "signal": "HOLD",
            "score": 0,
            "reasons": [
                "Not enough market data"
            ]
        }

    latest = df.iloc[-1]

    score = 0
    reasons = []

    close = latest["Close"]
    ma20 = latest["MA20"]
    ma50 = latest["MA50"]
    ma200 = latest["MA200"]
    rsi = latest["RSI"]
    macd = latest["MACD"]
    signal_line = latest["Signal"]
    bb_upper = latest["BB_Upper"]
    bb_lower = latest["BB_Lower"]

    # Moving averages
    if pd.notna(ma20):

        if close > ma20:

            score += 1
            reasons.append(
                "Price is above the 20-day moving average"
            )

        else:

            score -= 1
            reasons.append(
                "Price is below the 20-day moving average"
            )

    if pd.notna(ma50):

        if close > ma50:

            score += 1
            reasons.append(
                "Price is above the 50-day moving average"
            )

        else:

            score -= 1
            reasons.append(
                "Price is below the 50-day moving average"
            )

    if pd.notna(ma200):

        if close > ma200:

            score += 1
            reasons.append(
                "Price is above the 200-day moving average"
            )

        else:

            score -= 1
            reasons.append(
                "Price is below the 200-day moving average"
            )

    # RSI
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
                "RSI is above 50"
            )

        else:

            score -= 1
            reasons.append(
                "RSI is below 50"
            )

    # MACD
    if pd.notna(macd) and pd.notna(signal_line):

        if macd > signal_line:

            score += 2
            reasons.append(
                "MACD is above the signal line"
            )

        else:

            score -= 2
            reasons.append(
                "MACD is below the signal line"
            )

    # Bollinger
    if (
        pd.notna(bb_upper)
        and pd.notna(bb_lower)
    ):

        if close < bb_lower:

            score += 1
            reasons.append(
                "Price is below the lower Bollinger Band"
            )

        elif close > bb_upper:

            score -= 1
            reasons.append(
                "Price is above the upper Bollinger Band"
            )

    if score >= 4:

        signal = "BUY"

    elif score <= -4:

        signal = "SELL"

    else:

        signal = "HOLD"

    return {
        "signal": signal,
        "score": score,
        "reasons": reasons
    }


@st.cache_data(
    ttl=60,
    show_spinner=False
)
def get_daily_data(ticker):

    try:

        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        return flatten_columns(data)

    except Exception:

        return pd.DataFrame()


@st.cache_data(
    ttl=30,
    show_spinner=False
)
def get_latest_market_data(ticker):

    try:

        data = yf.download(
            ticker,
            period="1d",
            interval="1m",
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False
        )

        data = flatten_columns(data)

        if data.empty:
            return None

        close = get_close_series(data)

        if close.empty:
            return None

        return {
            "price": float(close.iloc[-1]),
            "timestamp": close.index[-1]
        }

    except Exception:

        return None


def format_timestamp(timestamp):

    try:

        ts = pd.Timestamp(timestamp)

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        ts = ts.tz_convert(IST)

        return ts.strftime(
            "%d %b %Y • %I:%M:%S %p IST"
        )

    except Exception:

        return "Unavailable"


# ============================================================
# MAIN APPLICATION
# ============================================================

if not symbol:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:70px 20px;
            color:#94a3b8;
        ">

            <div style="font-size:55px;">
                📊
            </div>

            <h2 style="color:#f8fafc;">
                Start Your Analysis
            </h2>

            <p>
                Enter a stock symbol in the sidebar to view
                the latest price and technical analysis.
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
# TICKER
# ============================================================

if market == "NSE":

    ticker = f"{symbol}.NS"

else:

    ticker = f"{symbol}.BO"


# ============================================================
# FETCH DATA
# ============================================================

with st.spinner(
    f"Analyzing {symbol}..."
):

    live_data = get_latest_market_data(ticker)

    daily_data = get_daily_data(ticker)


if daily_data.empty:

    st.error(
        f"❌ Could not find data for {ticker}. "
        "Please verify the stock symbol."
    )

    st.stop()


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

analysis = calculate_indicators(
    daily_data
)


if analysis.empty:

    st.error(
        "Unable to calculate technical indicators."
    )

    st.stop()


signal_data = generate_signal(
    analysis
)

signal = signal_data["signal"]
score = signal_data["score"]
reasons = signal_data["reasons"]


latest = analysis.iloc[-1]


# ============================================================
# CURRENT PRICE
# ============================================================

technical_price = float(
    latest["Close"]
)


if live_data is not None:

    current_price = live_data["price"]
    market_timestamp = live_data["timestamp"]

else:

    current_price = technical_price
    market_timestamp = analysis.index[-1]


app_timestamp = datetime.now(
    IST
)


# ============================================================
# TOP PRICE + SIGNAL SECTION
# ============================================================

price_col, signal_col = st.columns(
    [1.55, 1],
    gap="large"
)


with price_col:

    st.markdown(
        f"""
        <div class="price-card">

            <div class="price-label">
                {symbol} • {market}
            </div>

            <div class="price-value">
                ₹{current_price:,.2f}
            </div>

            <div class="timestamp">
                Market data:
                {format_timestamp(market_timestamp)}
            </div>

            <div class="timestamp">
                App updated:
                {app_timestamp.strftime("%d %b %Y • %I:%M:%S %p IST")}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with signal_col:

    if signal == "BUY":

        signal_class = "signal-buy"
        signal_icon = "🟢"

    elif signal == "SELL":

        signal_class = "signal-sell"
        signal_icon = "🔴"

    else:

        signal_class = "signal-hold"
        signal_icon = "🟡"


    st.markdown(
        f"""
        <div class="signal-box {signal_class}">

            <div class="signal-title">
                Technical Signal
            </div>

            <div class="signal-value">
                {signal_icon} {signal}
            </div>

            <div class="signal-score">
                Technical Score: {score:+d}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MARKET DISCLAIMER
# ============================================================

st.caption(
    "⚠️ Latest available price from Yahoo Finance. "
    "Market data may be delayed depending on the data feed."
)


# ============================================================
# KEY INDICATORS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Key Technical Indicators</div>',
    unsafe_allow_html=True
)


rsi = latest["RSI"]
macd = latest["MACD"]
ma20 = latest["MA20"]
ma50 = latest["MA50"]
ma200 = latest["MA200"]


indicator_cols = st.columns(
    5,
    gap="medium"
)


def metric_card(
    label,
    value,
    description
):

    return f"""
    <div class="metric-card">

        <div class="metric-label">
            {label}
        </div>

        <div class="metric-value">
            {value}
        </div>

        <div class="metric-small">
            {description}
        </div>

    </div>
    """


with indicator_cols[0]:

    value = (
        f"{rsi:.1f}"
        if pd.notna(rsi)
        else "N/A"
    )

    st.markdown(
        metric_card(
            "RSI",
            value,
            "14-day momentum"
        ),
        unsafe_allow_html=True
    )


with indicator_cols[1]:

    value = (
        f"{macd:.2f}"
        if pd.notna(macd)
        else "N/A"
    )

    st.markdown(
        metric_card(
            "MACD",
            value,
            "Trend momentum"
        ),
        unsafe_allow_html=True
    )


with indicator_cols[2]:

    value = (
        f"₹{ma20:,.0f}"
        if pd.notna(ma20)
        else "N/A"
    )

    st.markdown(
        metric_card(
            "MA20",
            value,
            "Short-term trend"
        ),
        unsafe_allow_html=True
    )


with indicator_cols[3]:

    value = (
        f"₹{ma50:,.0f}"
        if pd.notna(ma50)
        else "N/A"
    )

    st.markdown(
        metric_card(
            "MA50",
            value,
            "Medium-term trend"
        ),
        unsafe_allow_html=True
    )


with indicator_cols[4]:

    value = (
        f"₹{ma200:,.0f}"
        if pd.notna(ma200)
        else "N/A"
    )

    st.markdown(
        metric_card(
            "MA200",
            value,
            "Long-term trend"
        ),
        unsafe_allow_html=True
    )


# ============================================================
# CHART
# ============================================================

st.markdown(
    '<div class="section-title">📈 Price Trend</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Current price compared with key moving averages'
    '</div>',
    unsafe_allow_html=True
)


chart_data = analysis.tail(180).copy()


fig, ax = plt.subplots(
    figsize=(14, 5)
)

fig.patch.set_facecolor(
    "#0f172a"
)

ax.set_facecolor(
    "#0f172a"
)

ax.plot(
    chart_data.index,
    chart_data["Close"],
    linewidth=2.5,
    label="Price"
)

ax.plot(
    chart_data.index,
    chart_data["MA20"],
    linewidth=1.5,
    label="MA20"
)

ax.plot(
    chart_data.index,
    chart_data["MA50"],
    linewidth=1.5,
    label="MA50"
)

if chart_data["MA200"].notna().any():

    ax.plot(
        chart_data.index,
        chart_data["MA200"],
        linewidth=1.5,
        label="MA200"
    )


ax.set_title(
    f"{symbol} — Price & Moving Averages",
    color="white",
    fontsize=14,
    fontweight="bold"
)

ax.tick_params(
    colors="#94a3b8"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color(
    "#334155"
)

ax.spines["bottom"].set_color(
    "#334155"
)

ax.grid(
    alpha=0.12
)

legend = ax.legend(
    frameon=False
)

for text in legend.get_texts():

    text.set_color(
        "#cbd5e1"
    )


plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Signal Analysis</div>',
    unsafe_allow_html=True
)


analysis_col1, analysis_col2 = st.columns(
    [1, 1],
    gap="large"
)


with analysis_col1:

    st.markdown(
        "#### Why this signal?"
    )

    for reason in reasons:

        st.markdown(
            f"• {reason}"
        )


with analysis_col2:

    st.markdown(
        "#### Market Interpretation"
    )

    if pd.notna(rsi):

        if rsi >= 70:

            st.write(
                "🔴 RSI indicates the stock may be overbought."
            )

        elif rsi <= 30:

            st.write(
                "🟢 RSI indicates the stock may be oversold."
            )

        else:

            st.write(
                "🟡 RSI is currently in a neutral range."
            )


    if (
        pd.notna(macd)
        and pd.notna(latest["Signal"])
    ):

        if macd > latest["Signal"]:

            st.write(
                "🟢 MACD momentum is currently bullish."
            )

        else:

            st.write(
                "🔴 MACD momentum is currently bearish."
            )


    if pd.notna(ma50):

        if current_price > ma50:

            st.write(
                "🟢 Price is trading above the 50-day trend."
            )

        else:

            st.write(
                "🔴 Price is trading below the 50-day trend."
            )


# ============================================================
# QUICK SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📋 Quick Summary</div>',
    unsafe_allow_html=True
)


summary_col1, summary_col2, summary_col3 = st.columns(
    3,
    gap="medium"
)


with summary_col1:

    if signal == "BUY":

        text = "Positive technical setup"

    elif signal == "SELL":

        text = "Negative technical setup"

    else:

        text = "Mixed technical setup"

    st.markdown(
        metric_card(
            "Overall View",
            signal,
            text
        ),
        unsafe_allow_html=True
    )


with summary_col2:

    if pd.notna(rsi):

        if rsi > 70:

            rsi_status = "Overbought"

        elif rsi < 30:

            rsi_status = "Oversold"

        else:

            rsi_status = "Neutral"

    else:

        rsi_status = "Unavailable"


    st.markdown(
        metric_card(
            "RSI Status",
            rsi_status,
            "Momentum condition"
        ),
        unsafe_allow_html=True
    )


with summary_col3:

    if pd.notna(ma50):

        if current_price > ma50:

            trend_status = "Bullish"

        else:

            trend_status = "Bearish"

    else:

        trend_status = "Unavailable"


    st.markdown(
        metric_card(
            "Trend",
            trend_status,
            "Based on MA50"
        ),
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>Indian Stock Analyzer</strong><br><br>

        Technical indicators are calculated using historical
        market data for analysis purposes.<br>

        Buy / Sell / Hold signals are algorithmic indicators
        and are not financial advice.<br><br>

        Data source: Yahoo Finance

    </div>
    """,
    unsafe_allow_html=True
)
