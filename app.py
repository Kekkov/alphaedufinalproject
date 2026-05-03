import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from groq import Groq
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Stock Analyst", layout="wide")
st.title("📉 Stock Predictor & AI Analyst")

# ─── Боковая панель ───────────────────────────────────────────────────────────
GROQ_API_KEY = st.sidebar.text_input("Groq API ключ", type="password")
ticker = st.sidebar.text_input("Введите тикер (например, TSLA, AAPL, BTC-USD)", "AAPL").strip().upper()
period = st.sidebar.slider("На сколько дней строим прогноз?", 7, 90, 30)

st.sidebar.divider()
st.sidebar.subheader("📊 Технические индикаторы")
show_bollinger = st.sidebar.checkbox("Полосы Боллинджера", value=True)
show_rsi       = st.sidebar.checkbox("RSI", value=True)
show_macd      = st.sidebar.checkbox("MACD", value=True)
show_ma        = st.sidebar.checkbox("Скользящие средние (MA20, MA50)", value=True)

st.sidebar.divider()
st.sidebar.subheader("🤖 Модели прогноза")
use_prophet = st.sidebar.checkbox("Prophet", value=True)
use_lr      = st.sidebar.checkbox("Linear Regression", value=True)
use_rf      = st.sidebar.checkbox("Random Forest", value=True)

# ─── Загрузка данных ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(symbol):
    df = yf.download(symbol, start="2022-01-01", end=datetime.now().strftime("%Y-%m-%d"), auto_adjust=True)
    if df.empty:
        return df, {}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    if pd.api.types.is_datetime64tz_dtype(df["Date"]):
        df["Date"] = df["Date"].dt.tz_localize(None)

    info = {}
    try:
        ticker_obj = yf.Ticker(symbol)
        raw = ticker_obj.info
        info = {
            "P/E":        raw.get("trailingPE", "N/A"),
            "EPS":        raw.get("trailingEps", "N/A"),
            "Market Cap": raw.get("marketCap", "N/A"),
            "Div Yield":  raw.get("dividendYield", "N/A"),
            "Beta":       raw.get("beta", "N/A"),
            "Sector":     raw.get("sector", "N/A"),
            "52w High":   raw.get("fiftyTwoWeekHigh", "N/A"),
            "52w Low":    raw.get("fiftyTwoWeekLow", "N/A"),
        }
    except Exception:
        pass
    return df, info

with st.spinner("Загрузка данных..."):
    df, fundamentals = load_data(ticker)

if df.empty:
    st.error("Данные не найдены. Проверьте правильность тикера.")
    st.stop()

# ─── Технические индикаторы ──────────────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    close = df["Close"]

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    df["BB_mid"]   = close.rolling(20).mean()
    df["BB_std"]   = close.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    return df

df = add_indicators(df)

# ─── ML модели ───────────────────────────────────────────────────────────────
def make_prophet_forecast(df_input, days):
    df_train = df_input[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df_train["ds"] = pd.to_datetime(df_train["ds"]).dt.tz_localize(None)
    df_train = df_train.dropna()
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df_train)
    future   = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)

def make_lr_forecast(df_input, days):
    close  = df_input["Close"].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)
    X      = np.arange(len(scaled)).reshape(-1, 1)
    model  = LinearRegression()
    model.fit(X, scaled)
    future_X   = np.arange(len(scaled), len(scaled) + days).reshape(-1, 1)
    prediction = scaler.inverse_transform(model.predict(future_X))
    last_date  = df_input["Date"].iloc[-1]
    dates      = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="B")
    return pd.DataFrame({"ds": dates, "yhat_lr": prediction.flatten()})

def make_rf_forecast(df_input, days):
    close  = df_input["Close"].dropna().values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.reshape(-1, 1)).flatten()
    window = 20
    X, y   = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i])
    X, y  = np.array(X), np.array(y)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    last_window = scaled[-window:].tolist()
    preds = []
    for _ in range(days):
        inp  = np.array(last_window[-window:]).reshape(1, -1)
        pred = model.predict(inp)[0]
        preds.append(pred)
        last_window.append(pred)
    preds     = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    last_date = df_input["Date"].iloc[-1]
    dates     = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="B")
    return pd.DataFrame({"ds": dates, "yhat_rf": preds})

prophet_fc = lr_fc = rf_fc = None
with st.spinner("Строим прогнозы..."):
    try:
        if use_prophet:
            prophet_fc = make_prophet_forecast(df, period)
        if use_lr:
            lr_fc = make_lr_forecast(df, period)
        if use_rf:
            rf_fc = make_rf_forecast(df, period)
    except Exception as e:
        st.error(f"Ошибка при построении прогноза: {e}")
        st.stop()

# ─── Фундаментальные показатели ──────────────────────────────────────────────
if fundamentals:
    st.subheader(f"📋 Фундаментальные показатели — {ticker}")
    cols  = st.columns(4)
    items = list(fundamentals.items())
    for i, (k, v) in enumerate(items):
        with cols[i % 4]:
            if k == "Market Cap" and isinstance(v, (int, float)):
                v = f"${v/1e9:.1f}B"
            elif k == "Div Yield" and isinstance(v, float):
                v = f"{v*100:.2f}%"
            elif isinstance(v, float):
                v = f"{v:.2f}"
            st.metric(k, v if v != "N/A" else "—")

# ─── Главный график ──────────────────────────────────────────────────────────
rows   = 1 + int(show_rsi) + int(show_macd)
heights = [0.6] + [0.2] * (rows - 1)
subplot_titles = [f"График и прогноз — {ticker}"]
if show_rsi:  subplot_titles.append("RSI (14)")
if show_macd: subplot_titles.append("MACD")

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    row_heights=heights, subplot_titles=subplot_titles,
                    vertical_spacing=0.05)

fig.add_trace(go.Candlestick(
    x=df["Date"], open=df["Open"], high=df["High"],
    low=df["Low"], close=df["Close"], name="Цена",
    increasing_line_color="limegreen", decreasing_line_color="tomato"
), row=1, col=1)

if show_ma:
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20",
                             line=dict(color="orange", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], name="MA50",
                             line=dict(color="violet", width=1.5)), row=1, col=1)

if show_bollinger:
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], name="BB Upper",
                             line=dict(color="gray", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], name="BB Lower",
                             line=dict(color="gray", dash="dot", width=1),
                             fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

if prophet_fc is not None:
    fig.add_trace(go.Scatter(x=prophet_fc["ds"], y=prophet_fc["yhat"],
                             name="Prophet", line=dict(color="firebrick", dash="dot", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([prophet_fc["ds"], prophet_fc["ds"][::-1]]),
        y=pd.concat([prophet_fc["yhat_upper"], prophet_fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(178,34,34,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Prophet интервал"
    ), row=1, col=1)

if lr_fc is not None:
    fig.add_trace(go.Scatter(x=lr_fc["ds"], y=lr_fc["yhat_lr"],
                             name="Linear Regression", line=dict(color="cyan", dash="dash", width=2)), row=1, col=1)

if rf_fc is not None:
    fig.add_trace(go.Scatter(x=rf_fc["ds"], y=rf_fc["yhat_rf"],
                             name="Random Forest", line=dict(color="gold", dash="dash", width=2)), row=1, col=1)

current_row = 2
if show_rsi:
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI",
                             line=dict(color="royalblue", width=1.5)), row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=current_row, col=1)
    current_row += 1

if show_macd:
    colors = ["limegreen" if v >= 0 else "tomato" for v in df["MACD_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_hist"], name="MACD Hist",
                         marker_color=colors), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD",
                             line=dict(color="royalblue", width=1.5)), row=current_row, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_signal"], name="Signal",
                             line=dict(color="orange", width=1.5)), row=current_row, col=1)

fig.update_layout(
    height=800, hovermode="x unified",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ─── AI Отчёт ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🤖 AI Аналитический отчёт (Groq)")

def generate_ai_analysis(ticker, history, fundamentals, period, prophet_fc, lr_fc, rf_fc, api_key):
    current_price = float(history["Close"].iloc[-1])
    rsi_val       = float(history["RSI"].iloc[-1])
    macd_val      = float(history["MACD"].iloc[-1])
    volatility    = float(history["Close"].tail(30).pct_change().std() * 100)

    rsi_status  = "(перекуплен)" if rsi_val > 70 else "(перепродан)" if rsi_val < 30 else "(нейтрален)"
    macd_status = "бычий сигнал" if macd_val > 0 else "медвежий сигнал"

    forecasts_text = ""
    if prophet_fc is not None:
        p = float(prophet_fc["yhat"].iloc[-1])
        pct = (p - current_price) / current_price * 100
        forecasts_text += f"- Prophet: {p:.2f} ({pct:+.2f}%)\n"
    if lr_fc is not None:
        p = float(lr_fc["yhat_lr"].iloc[-1])
        pct = (p - current_price) / current_price * 100
        forecasts_text += f"- Linear Regression: {p:.2f} ({pct:+.2f}%)\n"
    if rf_fc is not None:
        p = float(rf_fc["yhat_rf"].iloc[-1])
        pct = (p - current_price) / current_price * 100
        forecasts_text += f"- Random Forest: {p:.2f} ({pct:+.2f}%)\n"

    fund_text = "\n".join([f"- {k}: {v}" for k, v in fundamentals.items()]) if fundamentals else "Нет данных"

    prompt = f"""Ты опытный финансовый аналитик. Проанализируй акцию {ticker}.

Текущая цена: {current_price:.2f}

Прогнозы на {period} дней:
{forecasts_text}
Технические индикаторы:
- RSI (14): {rsi_val:.1f} {rsi_status}
- MACD: {macd_val:.3f} {macd_status}
- Волатильность (30д): {volatility:.2f}%

Фундаментальные показатели:
{fund_text}

Напиши структурированный отчёт на русском:
1. Тренд — бычий или медвежий, обоснование с учётом RSI и MACD
2. Согласованность моделей — совпадают ли прогнозы ML моделей
3. Риски — 2-3 ключевых риска
4. Рекомендация — Hold / Buy / Sell с обоснованием

Заверши дисклеймером, что это не инвестиционная рекомендация."""

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if st.button("Сгенерировать AI-отчёт"):
    if not GROQ_API_KEY:
        st.warning("Введите Groq API ключ в боковой панели!")
    else:
        with st.spinner("AI изучает графики..."):
            try:
                analysis = generate_ai_analysis(
                    ticker, df, fundamentals, period,
                    prophet_fc, lr_fc, rf_fc, GROQ_API_KEY
                )
                st.info(analysis)
            except Exception as e:
                st.error(f"Ошибка: {e}")

# ─── Сырые данные ─────────────────────────────────────────────────────────────
with st.expander("Посмотреть сырые данные"):
    st.dataframe(
        df[["Date", "Close", "MA20", "MA50", "BB_upper", "BB_lower", "RSI", "MACD"]].tail(20),
        use_container_width=True
    )
