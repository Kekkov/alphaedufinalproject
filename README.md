# 📉 AI Stock Analyst

<img width="1919" height="914" alt="image" src="https://github.com/user-attachments/assets/898c9116-6ec1-443b-9974-aed88a82c7fc" />
<img width="1919" height="911" alt="image" src="https://github.com/user-attachments/assets/05310ad2-503f-4b6e-b6e0-e7df983bf31e" />

Приложение для анализа и прогнозирования акций с использованием машинного обучения и AI-аналитики.

## Возможности

- 📈 Исторические данные акций через Yahoo Finance
- 🤖 Три ML-модели прогнозирования: Prophet, Linear Regression, Random Forest
- 📊 Технические индикаторы: Полосы Боллинджера, RSI, MACD, MA20/MA50
- 💼 Фундаментальные показатели: P/E, EPS, Market Cap, Beta и др.
- 🧠 AI-аналитический отчёт через Groq (LLaMA 3.3 70B)

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
streamlit run app.py
```

## Использование

1. Введи тикер акции в боковой панели (например, `AAPL`, `TSLA`, `BTC-USD`)
2. Выбери период прогноза (7–90 дней)
3. Включи нужные индикаторы и модели
4. Вставь Groq API ключ для генерации AI-отчёта
5. Нажми **"Сгенерировать AI-отчёт"**

## Получение Groq API ключа

Зарегистрируйся на [console.groq.com](https://console.groq.com) — ключ бесплатный.


## Дисклеймер

Приложение создано в образовательных целях. Не является инвестиционной рекомендацией.
