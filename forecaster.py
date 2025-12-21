import pandas as pd
from models import RandomForestModel, ArimaModel, LSTMModel

def train_and_predict(data):
    """
    Обучает три модели, сравнивает их RMSE и возвращает прогноз лучшей.
    """
    # Берем только серию цен
    series = data['Close']
    
    # Список кандидатов
    candidates = {
        "Random Forest": RandomForestModel(lags=15),
        "ARIMA": ArimaModel(),
        "LSTM (Neural Net)": LSTMModel(look_back=30)
    }
    
    results = {}
    
    print("--- Начало соревнования моделей ---")
    
    for name, model in candidates.items():
        try:
            print(f"⏳ Обучение {name}...")
            forecast, metrics = model.train_and_predict(series)
            
            results[name] = {
                'forecast': forecast,
                'metrics': metrics
            }
            print(f"✅ {name}: RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
            
        except Exception as e:
            print(f"❌ Ошибка в модели {name}: {e}")
            # Если модель упала, не добавляем её в результаты
            continue

    if not results:
        raise Exception("Все модели завершились с ошибкой. Проверьте данные.")

    # Выбор победителя (у кого меньше RMSE)
    best_model_name = min(results, key=lambda x: results[x]['metrics']['rmse'])
    
    best_result = results[best_model_name]
    print(f"🏆 Победитель: {best_model_name}")
    
    return best_model_name, best_result['forecast'], best_result['metrics']