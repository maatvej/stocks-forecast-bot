import numpy as np
import pandas as pd
import warnings
import logging
from scipy.stats import norm

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# Statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# Настройки
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# --- Вспомогательные функции ---

def evaluate_model(y_true, y_pred):
    """Расчет метрик RMSE и MAPE."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'rmse': 9999, 'mape': 9999}
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = 100.0
    return {'rmse': rmse, 'mape': mape}

def create_lags(series, lags=10):
    """Создает лаговые признаки для Random Forest."""
    df = pd.DataFrame(series)
    columns = [df.shift(i) for i in range(1, lags + 1)]
    df_lags = pd.concat(columns, axis=1)
    df_lags.columns = [f'lag_{i}' for i in range(1, lags + 1)]
    df_lags = df_lags.dropna()
    y = series.iloc[lags:]
    return df_lags.values, y.values

# --- Модели ---

class RandomForestModel:
    def __init__(self, lags=15):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.lags = lags

    def train_and_predict(self, series):
        X, y = create_lags(series, self.lags)
        
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.model.fit(X_train, y_train)
        test_pred = self.model.predict(X_test)
        metrics = evaluate_model(y_test, test_pred)
        
        # Финальное обучение на всех данных
        self.model.fit(X, y)
        
        # Рекурсивный прогноз
        future_forecast = []
        last_window = series.values[-self.lags:]
        
        for _ in range(30):
            pred = self.model.predict(last_window.reshape(1, -1))[0]
            future_forecast.append(pred)
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred
            
        dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=30)
        return pd.Series(future_forecast, index=dates), metrics

# ==============================================================================
# ОБНОВЛЕННАЯ ARIMA (С УМНЫМ ПОДБОРОМ p, d, q)
# ==============================================================================
class ArimaModel:
    def _make_stationary(self, series, max_diff=2):
        """
        Определяет порядок интегрирования d с помощью теста Дики-Фуллера.
        """
        d = 0
        series_pd = pd.Series(series).dropna()
        
        if len(series_pd) < 10:
            return series_pd, 0

        # Тест на стационарность
        try:
            p_value = adfuller(series_pd)[1]
        except:
            return series_pd, 1 # Fallback

        while p_value > 0.05 and d < max_diff:
            d += 1
            series_pd = series_pd.diff().dropna()
            if len(series_pd) < 10:
                 return pd.Series(series), d-1
            try:
                p_value = adfuller(series_pd)[1]
            except:
                break
        
        return series_pd, d

    def _determine_p_q(self, series, alpha=0.05, max_lag=26, max_p=5, max_q=5, tolerance_factor=1.1):
        """
        Определяет p и q через ACF/PACF с учетом 'буфера' (tolerance_factor).
        """
        N = len(series)
        if N < 10:
            return 1, 1

        # Порог значимости
        significance_threshold = norm.ppf(1 - alpha / 2) / np.sqrt(N)
        tolerant_threshold = significance_threshold * tolerance_factor
        
        nlags = min(max_lag, N // 2 - 1)

        try:
            acf_values = acf(series, nlags=nlags, fft=False)
            pacf_values = pacf(series, nlags=nlags, method='ywm')
        except:
            return 1, 1

        # Подсчет значимых лагов
        p = sum(abs(val) > tolerant_threshold for val in pacf_values[1:])
        q = sum(abs(val) > tolerant_threshold for val in acf_values[1:])
        
        p = min(p, max_p)
        q = min(q, max_q)
        
        # Для акций часто лучше иметь хотя бы минимальную авторегрессию
        return max(p, 1), max(q, 1)

    def train_and_predict(self, series):
        # 1. Разделение данных
        split = int(len(series) * 0.9)
        train, test = series[:split], series[split:]
        
        # 2. Автоматический подбор параметров на тренировочной выборке
        # Сначала ищем d
        stationary_train, d = self._make_stationary(train)
        # Затем ищем p и q на стационарном ряду
        p, q = self._determine_p_q(stationary_train)
        
        print(f"  -> ARIMA Auto-params: p={p}, d={d}, q={q}")

        # 3. Обучение модели для оценки (Train/Test)
        try:
            # Для акций сезонность (seasonal_order) обычно отключают или ставят 0,
            # так как недельная сезонность (5 дней) сложна для простой SARIMA.
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(0, 0, 0, 0), 
                           enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=50)
            
            start = len(train)
            end = len(train) + len(test) - 1
            test_pred = res.predict(start=start, end=end)
            metrics = evaluate_model(test, test_pred)
        except Exception as e:
            print(f"ARIMA training error: {e}")
            metrics = {'rmse': 9999, 'mape': 9999}

        # 4. Финальное обучение на ВСЕХ данных с найденными параметрами
        try:
            final_model = SARIMAX(series, order=(p, d, q), seasonal_order=(0, 0, 0, 0),
                                 enforce_stationarity=False, enforce_invertibility=False)
            final_res = final_model.fit(disp=False, maxiter=50)
            
            forecast = final_res.get_forecast(steps=30).predicted_mean
        except:
            # Если финальное обучение упало, возвращаем пустой прогноз (или fallback)
            forecast = pd.Series([series.iloc[-1]]*30, index=pd.date_range(start=series.index[-1], periods=30))

        return forecast, metrics

# ==============================================================================
# LSTM (Без изменений)
# ==============================================================================
class LSTMModel:
    def __init__(self, look_back=30):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _create_dataset(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.look_back):
            X.append(dataset[i:(i + self.look_back), 0])
            Y.append(dataset[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def train_and_predict(self, series):
        data = series.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = self._create_dataset(scaled_data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = Sequential([
            LSTM(50, input_shape=(self.look_back, 1)),
            Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)
        
        test_pred_scaled = model.predict(X_test, verbose=0)
        test_pred = self.scaler.inverse_transform(test_pred_scaled)
        y_test_real = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        metrics = evaluate_model(y_test_real.flatten(), test_pred.flatten())
        
        future_forecast = []
        last_batch = scaled_data[-self.look_back:]
        
        for _ in range(30):
            input_data = last_batch.reshape(1, self.look_back, 1)
            pred_scaled = model.predict(input_data, verbose=0)
            pred_val = pred_scaled[0, 0]
            future_forecast.append(pred_val)
            last_batch = np.roll(last_batch, -1)
            last_batch[-1] = pred_val
            
        future_forecast = self.scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))
        dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=30)
        return pd.Series(future_forecast.flatten(), index=dates), metrics