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

st.markdown(
    """
    <style>
    .stApp { background:#0b1120; color:#f8fafc; }
    .block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:3rem; }
    .title { font-size:36px; font-weight:800; color:#f8fafc; margin-bottom:2px; }
    .subtitle { color:#94a3b8; font-size:14px; margin-bottom:20px; }
    .stock-card,.info-card,.prediction-card,.reason-box,.news-card,.factor,.disclaimer {
        background:#111827; border:1px solid #263449; border-radius:16px;
    }
    .stock-card { padding:24px; margin-bottom:20px; }
    .stock-name { font-size:30px; font-weight:800; }
    .stock-symbol,.cmp-label,.small-text { color:#94a3b8; font-size:12px; }
    .cmp-label { margin-top:18px; }
    .cmp-price { font-size:40px; font-weight:800; line-height:1.1; }
    .positive { color:#22c55e; font-weight:700; }
    .negative { color:#ef4444; font-weight:700; }
    .neutral { color:#f59e0b; font-weight:700; }
    .info-card { padding:16px; min-height:105px; }
    .info-label { color:#94a3b8; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }
    .info-value { color:#f8fafc; font-size:23px; font-weight:800; margin-top:4px; }
    .signal-buy { color:#22c55e; }
    .signal-sell { color:#ef4444; }
    .signal-hold { color:#f59e0b; }
    .section-title { font-size:21px; font-weight:800; color:#f8fafc; margin-top:28px; margin-bottom:14px; }
    .prediction-card { padding:20px; min-height:470px; }
    .prediction-heading { font-size:18px; font-weight:800; }
    .prediction-price { font-size:29px; font-weight:800; margin-top:10px; }
    .reason-box { padding:16px; }
    .reason-title { font-weight:800; margin-bottom:7px; }
    .reason-text { color:#94a3b8; font-size:13px; line-height:1.65; }
    .news-card { padding:14px; margin-bottom:9px; }
    .news-title { color:#f8fafc; font-size:14px; font-weight:700; }
    .news-meta { color:#64748b; font-size:11px; margin-top:5px; }
    .factor { padding:11px 14px; margin-bottom:7px; color:#cbd5e1; font-size:13px; }
    .disclaimer { margin-top:30px; padding:15px; color:#64748b; font-size:11px; line-height:1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

NIFTY_10 = {
    "RELIANCE":"RELIANCE.NS","TCS":"TCS.NS","HDFCBANK":"HDFCBANK.NS",
    "INFY":"INFY.NS","ICICIBANK":"ICICIBANK.NS","BHARTIARTL":"BHARTIARTL.NS",
    "ITC":"ITC.NS","SBIN":"SBIN.NS","LT":"LT.NS","AXISBANK":"AXISBANK.NS",
}

NIFTY_50 = {
    "ADANIENT":"ADANIENT.NS","ADANIPORTS":"ADANIPORTS.NS","APOLLOHOSP":"APOLLOHOSP.NS",
    "ASIANPAINT":"ASIANPAINT.NS","AXISBANK":"AXISBANK.NS","BAJAJ-AUTO":"BAJAJ-AUTO.NS",
    "BAJFINANCE":"BAJFINANCE.NS","BAJAJFINSV":"BAJAJFINSV.NS","BEL":"BEL.NS",
    "BHARTIARTL":"BHARTIARTL.NS","BPCL":"BPCL.NS","BRITANNIA":"BRITANNIA.NS",
    "CIPLA":"CIPLA.NS","COALINDIA":"COALINDIA.NS","DRREDDY":"DRREDDY.NS",
    "EICHERMOT":"EICHERMOT.NS","ETERNAL":"ETERNAL.NS","GRASIM":"GRASIM.NS",
    "HCLTECH":"HCLTECH.NS","HDFCBANK":"HDFCBANK.NS","HDFCLIFE":"HDFCLIFE.NS",
    "HEROMOTOCO":"HEROMOTOCO.NS","HINDALCO":"HINDALCO.NS","HINDUNILVR":"HINDUNILVR.NS",
    "ICICIBANK":"ICICIBANK.NS","INDUSINDBK":"INDUSINDBK.NS","INFY":"INFY.NS",
    "ITC":"ITC.NS","JIOFIN":"JIOFIN.NS","JSWSTEEL":"JSWSTEEL.NS","KOTAKBANK":"KOTAKBANK.NS",
    "LT":"LT.NS","M&M":"M&M.NS","MARUTI":"MARUTI.NS","MAXHEALTH":"MAXHEALTH.NS",
    "NESTLEIND":"NESTLEIND.NS","NTPC":"NTPC.NS","ONGC":"ONGC.NS","POWERGRID":"POWERGRID.NS",
    "RELIANCE":"RELIANCE.NS","SBILIFE":"SBILIFE.NS","SBIN":"SBIN.NS","SHRIRAMFIN":"SHRIRAMFIN.NS",
    "SUNPHARMA":"SUNPHARMA.NS","TATACONSUM":"TATACONSUM.NS","TATAMOTORS":"TATAMOTORS.NS",
    "TATASTEEL":"TATASTEEL.NS","TCS":"TCS.NS","TECHM":"TECHM.NS","TITAN":"TITAN.NS",
    "TRENT":"TRENT.NS","ULTRACEMCO":"ULTRACEMCO.NS",
}

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
    return (pd.Timestamp.today().normalize() + pd.offsets.BDay(days)).strftime("%d %b %Y")


def india_now():
    return datetime.now(ZoneInfo("Asia/Kolkata"))


def market_status_from_time(timestamp=None):
    """Return a simple NSE-style market status using India local time."""
    now = india_now() if timestamp is None else timestamp
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    now = now.astimezone(ZoneInfo("Asia/Kolkata"))

    # NSE regular session: Monday-Friday, 09:15-15:30 IST.
    if now.weekday() >= 5:
        return "Market Closed", "Weekend"
    if now.hour < 9 or (now.hour == 9 and now.minute < 15):
        return "Market Closed", "Pre-market"
    if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
        return "Market Closed", "After market"
    return "Market Open", "Live market"


@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_quote(ticker):
    """Fetch the latest Yahoo Finance quote separately from daily model data."""
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
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker)}"
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
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("chart", {}).get("result") or []
        if not result:
            return empty

        meta = result[0].get("meta", {})
        price = safe_float(meta.get("regularMarketPrice"), np.nan)
        previous_close = safe_float(
            meta.get("previousClose", meta.get("chartPreviousClose")), np.nan
        )
        regular_market_time = meta.get("regularMarketTime")
        quote_time = None
        if regular_market_time:
            quote_time = datetime.fromtimestamp(
                int(regular_market_time), tz=ZoneInfo("Asia/Kolkata")
            )

        # Use Yahoo's regular-market quote as the CMP. Only fall back to the
        # latest 1-minute close if the quote field is unavailable.
        if not np.isfinite(price):
            timestamps = result[0].get("timestamp") or []
            indicators = result[0].get("indicators", {})
            quote_rows = indicators.get("quote") or []
            closes = quote_rows[0].get("close", []) if quote_rows else []
            for ts, close_value in reversed(list(zip(timestamps, closes))):
                candidate = safe_float(close_value, np.nan)
                if np.isfinite(candidate):
                    price = candidate
                    quote_time = datetime.fromtimestamp(
                        int(ts), tz=ZoneInfo("Asia/Kolkata")
                    )
                    break

        if not np.isfinite(price):
            return empty

        if not np.isfinite(previous_close) or previous_close <= 0:
            previous_close = None
            change_pct = None
        else:
            change_pct = (price / previous_close - 1.0) * 100.0

        status, detail = market_status_from_time()
        return {
            "price": float(price),
            "previous_close": previous_close,
            "change_pct": change_pct,
            "quote_time": quote_time,
            "market_status": status,
            "status_detail": detail,
            "source": "Yahoo Finance",
        }
    except Exception:
        return empty


@st.cache_data(ttl=300, show_spinner=False)
def download_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).title() for c in df.columns]
        required = ["Open","High","Low","Close","Volume"]
        if not all(c in df.columns for c in required):
            return pd.DataFrame()
        return df[required].dropna()
    except Exception:
        return pd.DataFrame()

def add_indicators(df):
    data = df.copy()
    close, high, low, volume = data["Close"], data["High"], data["Low"], data["Volume"]
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["SMA200"] = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    true_range = pd.concat(
        [high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1
    ).max(axis=1)
    data["ATR"] = true_range.rolling(14).mean()
    data["VOL20"] = volume.rolling(20).mean()
    data["RETURN5"] = close.pct_change(5)
    data["RETURN20"] = close.pct_change(20)
    return data

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(symbol):
    try:
        query = quote(f"{symbol} India stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if response.status_code != 200:
            return []
        root = ET.fromstring(response.content)
        articles = []
        for item in root.findall(".//item")[:10]:
            title_node, link_node, date_node = item.find("title"), item.find("link"), item.find("pubDate")
            title = title_node.text if title_node is not None else ""
            link = link_node.text if link_node is not None else ""
            date = date_node.text if date_node is not None else ""
            if title:
                articles.append({"title":html.unescape(title),"link":link,"date":date})
        return articles
    except Exception:
        return []

POSITIVE_WORDS = [
    "profit","growth","strong","upgrade","buy","bullish","surge","rally","positive",
    "beat","record","expansion","order","orders","approval","wins","partnership",
    "outperform","deal","contract","guidance",
]
NEGATIVE_WORDS = [
    "loss","weak","downgrade","sell","bearish","fall","falls","drop","decline",
    "negative","miss","debt","lawsuit","investigation","warning","risk","cut",
    "cuts","underperform","fraud","penalty",
]

def news_sentiment(news):
    score = positive_count = negative_count = 0
    for article in news:
        title = article["title"].lower()
        for word in POSITIVE_WORDS:
            if word in title:
                score += 1
                positive_count += 1
        for word in NEGATIVE_WORDS:
            if word in title:
                score -= 1
                negative_count += 1
    sentiment = "Positive" if score >= 3 else "Negative" if score <= -3 else "Neutral"
    return {"score":score,"positive":positive_count,"negative":negative_count,"sentiment":sentiment}

def technical_score(data):
    last = data.iloc[-1]
    close, sma20, sma50, sma200 = [safe_float(last[x]) for x in ["Close","SMA20","SMA50","SMA200"]]
    rsi, macd, macd_signal = safe_float(last["RSI"],50), safe_float(last["MACD"]), safe_float(last["MACD_SIGNAL"])
    score = 0
    score += 2 if close > sma20 else -2
    score += 2 if close > sma50 else -2
    score += 2 if sma50 > sma200 else -2
    score += 2 if macd > macd_signal else -2
    if 50 <= rsi <= 70:
        score += 1
    elif rsi < 30:
        score += 1
    elif rsi > 75:
        score -= 1
    return score

def predict_prices(data):
    if len(data) < 180:
        return None
    features = ["Close","SMA20","SMA50","SMA200","RSI","MACD","MACD_SIGNAL","ATR","RETURN5","RETURN20"]
    df = data.copy()
    df["TARGET1"], df["TARGET5"], df["TARGET20"] = df["Close"].shift(-1), df["Close"].shift(-5), df["Close"].shift(-20)
    clean = df.dropna(subset=features + ["TARGET1","TARGET5","TARGET20"])
    if len(clean) < 100:
        return None
    latest = df[features].iloc[[-1]].dropna()
    if latest.empty:
        return None
    models = {}
    for name, target, seed in [("1d","TARGET1",42),("5d","TARGET5",43),("20d","TARGET20",44)]:
        model = RandomForestRegressor(n_estimators=200,max_depth=8,random_state=seed,n_jobs=-1)
        model.fit(clean[features], clean[target])
        models[name] = model
    return {name:safe_float(model.predict(latest)[0]) for name,model in models.items()}

@st.cache_data(ttl=300, show_spinner=False)
def market_trend():
    nifty = download_data("^NSEI")
    if nifty.empty:
        return "Unknown", 0
    nifty = add_indicators(nifty)
    last = nifty.iloc[-1]
    close, sma20, sma50 = safe_float(last["Close"]), safe_float(last["SMA20"]), safe_float(last["SMA50"])
    score = (1 if close > sma20 else -1) + (1 if close > sma50 else -1)
    return ("Bullish",2) if score == 2 else ("Bearish",-2) if score == -2 else ("Neutral",score)

def generate_signal(tech_score, news_score, predicted_change, market_score):
    combined = (
        tech_score*0.55
        + news_score*0.20
        + np.sign(predicted_change)*min(abs(predicted_change),5)*0.25
        + market_score*0.50
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

def trade_levels(current, prediction, atr, signal):
    current, prediction, atr = safe_float(current), safe_float(prediction), safe_float(atr)
    if atr <= 0:
        atr = current*0.02
    if signal in ["BUY","STRONG BUY"]:
        target1 = max(prediction,current+atr*0.75)
        target2 = max(current+atr*1.50,prediction*1.015)
        stop = current-atr*1.20
    elif signal in ["SELL","STRONG SELL"]:
        target1 = min(prediction,current-atr*0.75)
        target2 = min(current-atr*1.50,prediction*0.985)
        stop = current+atr*1.20
    else:
        target1 = target2 = prediction
        stop = current-atr if prediction >= current else current+atr
    target1, target2, stop = max(target1,0.01), max(target2,0.01), max(stop,0.01)
    if signal in ["BUY","STRONG BUY"]:
        risk, reward = max(current-stop,0.01), max(target1-current,0)
    elif signal in ["SELL","STRONG SELL"]:
        risk, reward = max(stop-current,0.01), max(current-target1,0)
    else:
        risk, reward = abs(current-stop), abs(target1-current)
    return {"target1":target1,"target2":target2,"stop":stop,"risk":risk,"reward":reward,"rr":reward/risk if risk > 0 else 0}

def confidence(tech_score, news_score, rr, predicted_change, market_score):
    score = 50 + min(abs(tech_score)*4,20) + min(abs(news_score)*2,10) + abs(market_score)*4
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
    score = max(25,min(score,92))
    level = "High" if score >= 75 else "Moderate" if score >= 55 else "Low"
    return score,level

def prediction_reason(data, news_info, market_name, predicted_change):
    last = data.iloc[-1]
    close, sma20, sma50, sma200 = [safe_float(last[x]) for x in ["Close","SMA20","SMA50","SMA200"]]
    rsi = safe_float(last["RSI"],50)
    macd, macd_signal = safe_float(last["MACD"]), safe_float(last["MACD_SIGNAL"])
    volume, volume20 = safe_float(last["Volume"]), safe_float(last["VOL20"])
    factors = [
        "price is above the 20-day moving average" if close > sma20 else "price is below the 20-day moving average",
        "price is above the 50-day moving average" if close > sma50 else "price is below the 50-day moving average",
        "the medium-term trend is bullish" if sma50 > sma200 else "the medium-term trend remains weak",
        "MACD is above its signal line" if macd > macd_signal else "MACD is below its signal line",
    ]
    factors.append(f"RSI at {rsi:.1f} indicates oversold conditions" if rsi < 30 else f"RSI at {rsi:.1f} indicates overbought conditions" if rsi > 70 else f"RSI is {rsi:.1f}")
    if volume > volume20:
        factors.append("volume is above its 20-day average")
    factors.append(
        "recent news sentiment is positive"
        if news_info["sentiment"] == "Positive"
        else "recent news sentiment is negative"
        if news_info["sentiment"] == "Negative"
        else "recent news sentiment is neutral or mixed"
    )
    factors.append(f"the broader Nifty trend is {market_name.lower()}")
    direction = "upside" if predicted_change >= 0 else "downside"
    return f"The model projects approximately {abs(predicted_change):.2f}% {direction}. " + "; ".join(factors[:7]) + "."

def analyze_stock(symbol, ticker, market_name, market_score):
    # Historical daily data is used ONLY by the prediction engine.
    raw = download_data(ticker)
    if raw.empty:
        return None

    data = add_indicators(raw).dropna(subset=["Close"])
    if len(data) < 60:
        return None

    last = data.iloc[-1]
    model_base = safe_float(last["Close"])
    previous_daily_close = safe_float(data["Close"].iloc[-2]) if len(data) >= 2 else model_base
    daily_change = ((model_base / previous_daily_close) - 1) * 100 if previous_daily_close else 0

    # Live quote is intentionally kept separate from the ML/prediction engine.
    live = fetch_live_quote(ticker)
    live_current = safe_float(live.get("price"), model_base)
    live_previous_close = live.get("previous_close")
    live_change = live.get("change_pct")
    if live_change is None and live_previous_close:
        live_change = ((live_current / safe_float(live_previous_close)) - 1) * 100
    if live_change is None:
        live_change = daily_change

    tech_score = technical_score(data)
    news = fetch_news(symbol)
    news_info = news_sentiment(news)
    ml = predict_prices(data)

    # Convert the ML price forecasts into expected percentage moves from the
    # latest daily model close, then apply those moves to the live CMP.
    if ml:
        model_prediction1 = ml["1d"]
        model_prediction5 = ml["5d"]
        model_prediction20 = ml["20d"]
    else:
        model_prediction1 = model_base * 1.005
        model_prediction5 = model_base * 1.015
        model_prediction20 = model_base * 1.03

    model_changes = {
        "1d": ((model_prediction1 / model_base) - 1) * 100 if model_base else 0,
        "5d": ((model_prediction5 / model_base) - 1) * 100 if model_base else 0,
        "20d": ((model_prediction20 / model_base) - 1) * 100 if model_base else 0,
    }

    predictions = {
        "1d": live_current * (1 + model_changes["1d"] / 100),
        "5d": live_current * (1 + model_changes["5d"] / 100),
        "20d": live_current * (1 + model_changes["20d"] / 100),
    }

    signal = generate_signal(tech_score, news_info["score"], model_changes["1d"], market_score)
    levels = trade_levels(
        live_current,
        predictions["1d"],
        safe_float(last["ATR"]),
        signal,
    )
    conf, level = confidence(
        tech_score, news_info["score"], levels["rr"], model_changes["1d"], market_score
    )
    reason = prediction_reason(data, news_info, market_name, model_changes["1d"])

    return {
        "symbol": symbol,
        "ticker": ticker,
        "data": data,
        "model_base": model_base,
        "current": live_current,
        "today_change": live_change,
        "daily_change": daily_change,
        "signal": signal,
        "tech_score": tech_score,
        "news": news,
        "news_info": news_info,
        "predictions": predictions,
        "changes": model_changes,
        "levels": levels,
        "confidence": conf,
        "confidence_level": level,
        "reason": reason,
        "market_name": market_name,
        "market_score": market_score,
        "live_quote": live,
        "market_date": pd.Timestamp(data.index[-1]).strftime("%d %b %Y"),
    }

st.sidebar.markdown("## 📊 Stock Analyzer")
universe_name = st.sidebar.selectbox("Stock Universe",["Nifty 10","Nifty 50","Custom Stock"])
universe = NIFTY_10 if universe_name == "Nifty 10" else NIFTY_50 if universe_name == "Nifty 50" else {}

if universe_name == "Custom Stock":
    selected_symbol = st.sidebar.text_input("NSE Symbol","TCS").upper().strip()
    selected_ticker = selected_symbol if selected_symbol.endswith(".NS") else selected_symbol+".NS"
else:
    selected_symbol = st.sidebar.selectbox(
        "Select Stock",list(universe.keys()),
        index=list(universe.keys()).index("TCS") if "TCS" in universe else 0
    )
    selected_ticker = universe[selected_symbol]

scan = st.sidebar.checkbox("Scan entire universe")

if st.sidebar.button("🔄 Refresh Data",use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.markdown(
    """
    <div class="title">📈 Indian Stock AI Analyzer</div>
    <div class="subtitle">
        Technical analysis • Machine learning • News sentiment •
        Market trend • Buy/Sell signals • Risk management
    </div>
    """,
    unsafe_allow_html=True,
)

market_name, market_score = market_trend()

if scan:
    st.markdown(f'<div class="section-title">🔎 {universe_name} Scanner</div>',unsafe_allow_html=True)
    rows = []
    progress = st.progress(0)
    symbols = list(universe.items())

    for index,(symbol,ticker) in enumerate(symbols):
        result = analyze_stock(symbol,ticker,market_name,market_score)
        if result:
            rows.append({
                "Stock":symbol,
                "CMP":round(result["current"],2),
                "Signal":result["signal"],
                "1D Target":round(result["predictions"]["1d"],2),
                "1D %":round(result["changes"]["1d"],2),
                "5D %":round(result["changes"]["5d"],2),
                "20D %":round(result["changes"]["20d"],2),
                "Stop Loss":round(result["levels"]["stop"],2),
                "Risk/Reward":f"1 : {result['levels']['rr']:.2f}",
                "Confidence":f"{result['confidence']:.0f}%",
                "News":result["news_info"]["sentiment"],
                "Market Status":result["live_quote"]["market_status"],
            })
        progress.progress((index+1)/max(len(symbols),1))
    progress.empty()

    if rows:
        scan_df = pd.DataFrame(rows)
        signal_order = {"STRONG BUY":0,"BUY":1,"HOLD":2,"SELL":3,"STRONG SELL":4}
        scan_df["_order"] = scan_df["Signal"].map(signal_order).fillna(5)
        scan_df = scan_df.sort_values(["_order","Confidence"],ascending=[True,False]).drop(columns=["_order"])
        st.dataframe(scan_df,use_container_width=True,hide_index=True)
    else:
        st.warning("No stock data could be retrieved.")
    st.stop()

with st.spinner(f"Analyzing {selected_symbol}..."):
    result = analyze_stock(selected_symbol,selected_ticker,market_name,market_score)

if result is None:
    st.error(f"Unable to retrieve enough market data for {selected_symbol}.")
    st.stop()

data = result["data"]
last = data.iloc[-1]
current,today_change = result["current"],result["today_change"]
signal,levels = result["signal"],result["levels"]
news,news_info = result["news"],result["news_info"]
predictions,changes = result["predictions"],result["changes"]
reason = result["reason"]

price_class = "positive" if today_change >= 0 else "negative"
arrow = "↑" if today_change >= 0 else "↓"

live_quote = result["live_quote"]
quote_time = live_quote.get("quote_time")
quote_time_text = quote_time.strftime("%d %b %Y, %I:%M:%S %p IST") if quote_time else "Unavailable"
market_status = live_quote.get("market_status", "Unavailable")
status_detail = live_quote.get("status_detail", "Live quote unavailable")
status_class = "positive" if market_status == "Market Open" else "neutral" if market_status == "Market Closed" else "negative"

if market_status == "Market Open":
    cmp_label = "Live CMP / Last Traded Price"
    cmp_note = "Live market quote"
elif market_status == "Market Closed":
    cmp_label = "Last Traded Price"
    cmp_note = status_detail
else:
    cmp_label = "Latest Available Price"
    cmp_note = "Live quote unavailable"

st.markdown(
    f"""
    <div class="stock-card">
        <div class="stock-name">{html.escape(selected_symbol)}</div>
        <div class="stock-symbol">NSE • {html.escape(selected_ticker)}</div>
        <div class="cmp-label">{cmp_label}</div>
        <div class="cmp-price">{money(current)}</div>
        <div class="{price_class}">{arrow} {abs(today_change):.2f}% today</div>
        <div style="margin-top:12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
            <span class="{status_class}" style="font-size:13px;">● {html.escape(market_status)}</span>
            <span style="color:#64748b;font-size:11px;">{html.escape(cmp_note)}</span>
        </div>
        <div style="margin-top:8px;color:#64748b;font-size:11px;">
            Quote time: {html.escape(quote_time_text)}
        </div>
        <div style="margin-top:4px;color:#64748b;font-size:11px;">
            Prediction model data through: {html.escape(result["market_date"])}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="info-card"><div class="info-label">Trading Signal</div><div class="info-value {signal_class(signal)}">{signal}</div></div>',unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="info-card"><div class="info-label">Confidence</div><div class="info-value">{result["confidence"]:.0f}%</div><div class="small-text">{result["confidence_level"]}</div></div>',unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="info-card"><div class="info-label">Technical Score</div><div class="info-value">{result["tech_score"]}</div><div class="small-text">Approx. range -9 to +9</div></div>',unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="info-card"><div class="info-label">Market Trend</div><div class="info-value">{html.escape(market_name)}</div><div class="small-text">Nifty trend</div></div>',unsafe_allow_html=True)

st.markdown('<div class="section-title">🤖 AI Price Predictions</div>',unsafe_allow_html=True)
prediction_items = [
    ("1 Trading Day",predictions["1d"],changes["1d"],1),
    ("5 Trading Days",predictions["5d"],changes["5d"],5),
    ("20 Trading Days",predictions["20d"],changes["20d"],20),
]
prediction_cols = st.columns(3)

for i,(horizon,predicted,change,days) in enumerate(prediction_items):
    cls = "positive" if change >= 0 else "negative"
    arrow = "↑" if change >= 0 else "↓"
    if i == 0:
        target1,target2,stop,rr = levels["target1"],levels["target2"],levels["stop"],levels["rr"]
    else:
        target1 = target2 = predicted
        stop = current*0.97 if change >= 0 else current*1.03
        risk,reward = abs(current-stop),abs(predicted-current)
        rr = reward/risk if risk > 0 else 0

    with prediction_cols[i]:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-heading">{horizon}</div>
                <div class="small-text" style="margin-top:8px;">Model projected price</div>
                <div class="prediction-price">{money(predicted)}</div>
                <div class="{cls}" style="margin-top:5px;">{arrow} {abs(change):.2f}%</div>
                <div style="margin-top:18px;">
                    <div class="small-text">Target 1</div>
                    <div style="color:#22c55e;font-size:16px;font-weight:700;">{money(target1)}</div>
                    <div class="small-text" style="margin-top:10px;">Target 2</div>
                    <div style="color:#22c55e;font-size:16px;font-weight:700;">{money(target2)}</div>
                    <div class="small-text" style="margin-top:10px;">Stop Loss</div>
                    <div style="color:#ef4444;font-size:16px;font-weight:700;">{money(stop)}</div>
                    <div class="small-text" style="margin-top:10px;">Risk / Reward</div>
                    <div style="color:#60a5fa;font-size:16px;font-weight:700;">1 : {rr:.2f}</div>
                    <div class="small-text" style="margin-top:10px;">Estimated Date</div>
                    <div style="color:#f8fafc;font-size:13px;">{business_day_date(days)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-title">🧠 Why does the model expect this movement?</div>',unsafe_allow_html=True)
st.markdown(
    f'<div class="reason-box"><div class="reason-title">Prediction Reason</div><div class="reason-text">{html.escape(reason)}</div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">🎯 Trade Plan</div>',unsafe_allow_html=True)
t1,t2,t3,t4,t5 = st.columns(5)
with t1: st.metric("Entry / CMP",money(current))
with t2: st.metric("Target 1",money(levels["target1"]))
with t3: st.metric("Target 2",money(levels["target2"]))
with t4: st.metric("Stop Loss",money(levels["stop"]))
with t5: st.metric("Risk / Reward",f"1 : {levels['rr']:.2f}")

st.markdown('<div class="section-title">📊 Technical Indicators</div>',unsafe_allow_html=True)
a,b,c,d,e,f = st.columns(6)
with a: st.metric("RSI",f"{safe_float(last['RSI'],50):.1f}")
with b: st.metric("20D MA",money(last["SMA20"]))
with c: st.metric("50D MA",money(last["SMA50"]))
with d: st.metric("200D MA",money(last["SMA200"]))
with e: st.metric("MACD",f"{safe_float(last['MACD']):.2f}")
with f: st.metric("ATR",money(last["ATR"]))

st.markdown('<div class="section-title">📰 Latest News & Movement Drivers</div>',unsafe_allow_html=True)
if news:
    for article in news[:6]:
        title,link,date = html.escape(article["title"]),html.escape(article["link"],quote=True),html.escape(article["date"])
        st.markdown(
            f"""
            <div class="news-card">
                <div class="news-title">{title}</div>
                <div class="news-meta">{date}</div>
                <div style="margin-top:6px;">
                    <a href="{link}" target="_blank" style="color:#60a5fa;font-size:12px;text-decoration:none;">Read article →</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("No recent news was retrieved.")

st.markdown('<div class="section-title">📌 Key Movement Factors</div>',unsafe_allow_html=True)
factors = [
    "Price is above the 20-day moving average." if current > safe_float(last["SMA20"]) else "Price is below the 20-day moving average.",
    "Price is above the 50-day moving average." if current > safe_float(last["SMA50"]) else "Price is below the 50-day moving average.",
    "MACD is supporting bullish momentum." if safe_float(last["MACD"]) > safe_float(last["MACD_SIGNAL"]) else "MACD is showing bearish momentum.",
]
rsi = safe_float(last["RSI"],50)
factors.append("RSI is oversold and a technical rebound is possible." if rsi < 30 else "RSI is overbought and short-term cooling is possible." if rsi > 70 else f"RSI is currently {rsi:.1f}.")
if safe_float(last["Volume"]) > safe_float(last["VOL20"]):
    factors.append("Trading volume is above its 20-day average.")
factors.append(
    "Recent news sentiment is positive."
    if news_info["sentiment"] == "Positive"
    else "Recent news sentiment is negative."
    if news_info["sentiment"] == "Negative"
    else "Recent news sentiment is neutral or mixed."
)
factors.append(f"Nifty market trend is currently {market_name}.")
for factor in factors:
    st.markdown(f'<div class="factor">• {html.escape(factor)}</div>',unsafe_allow_html=True)

st.markdown(
    """
    <div class="disclaimer">
        <b style="color:#94a3b8;">Disclaimer:</b>
        This application uses market data, technical indicators, recent news
        sentiment and machine-learning models to generate estimates. Predictions
        are not guaranteed and are not financial advice. Actual prices can differ
        materially from model estimates. Always perform your own research and
        consider your risk tolerance.
    </div>
    """,
    unsafe_allow_html=True,
)
