import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def get_recommendations(forecast, initial_amount):
    """
    Анализирует прогноз, находит точки входа/выхода и считает прибыль.
    """
    series = forecast.values
    dates = forecast.index
    
    # 1. Поиск сигналов (локальные экстремумы)
    # distance=2 фильтрует слишком частые сделки (шум)
    buy_indices, _ = find_peaks(-series, distance=2)
    sell_indices, _ = find_peaks(series, distance=2)
    
    # Превращаем в списки для удобства
    buy_indices = list(buy_indices)
    sell_indices = list(sell_indices)
    
    strategy_type = "Свинговая торговля (по пикам)"

    # 2. ЗАПАСНОЙ ПЛАН: Если график гладкий и пиков нет
    if not buy_indices and not sell_indices:
        # Ищем просто минимум и максимум на всем отрезке
        min_idx = np.argmin(series)
        max_idx = np.argmax(series)
        
        # Если минимум раньше максимума — это тренд вверх
        if min_idx < max_idx:
            buy_indices = [min_idx]
            sell_indices = [max_idx]
            strategy_type = "Трендовая (купи и держи)"
        else:
            # Если максимум раньше минимума — это падение
            strategy_type = "Выжидание (нисходящий тренд)"

    # 3. Симуляция торговли
    # Нам нужно объединить сигналы в хронологическом порядке
    events = []
    for idx in buy_indices:
        events.append((idx, 'buy'))
    for idx in sell_indices:
        events.append((idx, 'sell'))
    
    # Сортируем события по времени (индексу)
    events.sort(key=lambda x: x[0])
    
    cash = float(initial_amount)
    shares = 0.0
    trade_log = []
    
    for idx, action in events:
        price = float(series[idx])
        date_str = dates[idx].strftime('%d.%m')
        
        if action == 'buy' and cash > 0:
            # Покупаем на все деньги
            shares = cash / price
            cash = 0
            trade_log.append(f"🟢 {date_str}: Покупка по ${price:.2f}")
            
        elif action == 'sell' and shares > 0:
            # Продаем все акции
            cash = shares * price
            shares = 0
            trade_log.append(f"🔴 {date_str}: Продажа по ${price:.2f}")

    # Если в конце остались акции, оцениваем их по последней цене
    final_balance = cash
    if shares > 0:
        last_price = float(series[-1])
        final_balance = shares * last_price
        trade_log.append(f"ℹ️ (Остаток акций оценен по ${last_price:.2f})")

    profit = final_balance - initial_amount
    
    # 4. Формирование отчета
    summary = f"📋 **Торговая стратегия:** {strategy_type}\n\n"
    
    if trade_log:
        summary += "**Рекомендуемые действия:**\n" + "\n".join(trade_log)
    else:
        summary += "Сигналов для прибыльной торговли не найдено."
        
    summary += f"\n\n💰 **Финансовый итог:**\n"
    summary += f"Начальный депозит: ${initial_amount:.2f}\n"
    summary += f"Итоговый баланс: ${final_balance:.2f}\n"
    
    if profit >= 0:
        summary += f"Прибыль: **+${profit:.2f}** 🤑"
    else:
        summary += f"Убыток: **${profit:.2f}** 📉"

    return summary, profit