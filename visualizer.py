import matplotlib
matplotlib.use('Agg')  # Обязательно: отключает интерактивное окно (нужно для сервера)
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_forecast(history, forecast, ticker):
    """
    Строит график: последние 6 месяцев истории + 30 дней прогноза.
    Сохраняет изображение и возвращает путь к файлу.
    """
    plt.figure(figsize=(12, 6))
    
    # Оставляем для графика только последние 120 торговых дней (~6 месяцев),
    # иначе линия прогноза будет слишком маленькой и незаметной.
    history_subset = history.iloc[-120:]
    
    # Рисуем историю (синяя линия)
    plt.plot(history_subset.index, history_subset['Close'], label='История (последние 6 мес.)', color='#1f77b4', linewidth=2)
    
    # Рисуем прогноз (красная пунктирная линия)
    plt.plot(forecast.index, forecast, label='Прогноз (30 дней)', color='#d62728', linestyle='--', linewidth=2)
    
    # Соединяем последнюю точку истории с первой точкой прогноза тонкой линией,
    # чтобы не было визуального разрыва
    last_hist_date = history_subset.index[-1]
    last_hist_price = history_subset['Close'].iloc[-1]
    first_pred_date = forecast.index[0]
    first_pred_price = forecast.iloc[0]
    plt.plot([last_hist_date, first_pred_date], [last_hist_price, first_pred_price], color='#d62728', linestyle='--', linewidth=2)

    # Оформление
    plt.title(f'Прогноз стоимости акций {ticker}', fontsize=14)
    plt.xlabel('Дата', fontsize=10)
    plt.ylabel('Цена (USD)', fontsize=10)
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Сохранение файла
    filename = f"forecast_{ticker}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Очищаем память
    
    return filename