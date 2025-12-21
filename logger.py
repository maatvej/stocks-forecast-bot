from datetime import datetime

LOG_FILE = 'logs.txt'

def log_request(user_id, ticker, amount, model_name, metrics, profit):
    """
    Сохраняет параметры запроса в текстовый файл.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Безопасное получение метрик (на случай, если ключа нет)
    rmse = metrics.get('rmse', 0.0)
    mape = metrics.get('mape', 0.0)
    
    # Формируем строку лога
    # Формат: Время | ID юзера | Тикер | Сумма | Модель | Ошибка MAPE | Прогноз прибыли
    log_entry = (
        f"{timestamp} | "
        f"User: {user_id} | "
        f"Ticker: {ticker} | "
        f"Sum: ${amount:.2f} | "
        f"Model: {model_name} | "
        f"RMSE: {rmse:.2f} | "
        f"MAPE: {mape:.2f}% | "
        f"Profit: ${profit:.2f}\n"
    )
    
    try:
        # 'a' (append) - добавляет запись в конец файла, не стирая старые
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"❌ Ошибка записи лога: {e}")