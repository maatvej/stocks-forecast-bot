import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def check_ticker_exists(ticker):
    """
    Проверяет, существует ли тикер, делая быстрый запрос за 1 день.
    Возвращает True, если данные есть.
    """
    try:
        stock = yf.Ticker(ticker)
        # Запрашиваем историю за 5 дней, чтобы исключить выходные/праздники
        hist = stock.history(period="5d")
        return not hist.empty
    except Exception:
        return False

def load_data(ticker):
    """
    Загружает данные за последние 2 года.
    Возвращает DataFrame только с колонкой 'Close'.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2*365)  # 2 года назад
    
    try:
        # auto_adjust=True корректирует цену с учетом дивидендов и сплитов (важно для ML)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty:
            return pd.DataFrame()

        # --- ОБРАБОТКА ФОРМАТА YFINANCE ---
        # Иногда yfinance возвращает MultiIndex (например, ('Close', 'AAPL')), иногда просто 'Close'
        
        # Если колонки имеют уровни (MultiIndex), избавляемся от уровня тикера
        if isinstance(data.columns, pd.MultiIndex):
            try:
                # Пытаемся получить уровень с названиями метрик (Close, Open...)
                data.columns = data.columns.get_level_values(0)
            except IndexError:
                pass

        # Проверяем наличие колонки Close
        if 'Close' in data.columns:
            df = data[['Close']].copy()
        elif 'Adj Close' in data.columns:
            df = data[['Adj Close']].rename(columns={'Adj Close': 'Close'}).copy()
        else:
            # Если ничего не подошло, берем первую колонку (обычно это Close)
            df = data.iloc[:, [0]].copy()
            df.columns = ['Close']
            
        # Удаляем пропуски, если они есть
        df = df.dropna()
        
        return df

    except Exception as e:
        print(f"Ошибка в load_data: {e}")
        return pd.DataFrame()