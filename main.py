import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackContext

# Импорт всех наших модулей
import data_handler
import forecaster
import visualizer
import analysis
import logger
from config import TELEGRAM_TOKEN

# Настройка логов консоли
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Этапы разговора
TICKER, AMOUNT = range(2)

async def start(update: Update, context: CallbackContext):
    """Начало диалога."""
    await update.message.reply_text(
        "👋 **Привет! Я AI-инвестор.**\n\n"
        "Я умею:\n"
        "1. 📥 Скачивать историю акций\n"
        "2. 🧠 Обучать ML-модели (LSTM, ARIMA, RF)\n"
        "3. 🔮 Строить прогноз на 30 дней\n"
        "4. 💡 Давать советы по покупке/продаже\n\n"
        "Введите **тикер компании** (например: AAPL, TSLA, BTC-USD):",
        parse_mode='Markdown'
    )
    return TICKER

async def get_ticker(update: Update, context: CallbackContext):
    """Получаем тикер и проверяем его."""
    ticker = update.message.text.upper().strip()
    
    await update.message.reply_text(f"🔎 Проверяю тикер {ticker}...")
    
    if not data_handler.check_ticker_exists(ticker):
        await update.message.reply_text(
            "❌ Тикер не найден или данные недоступны.\n"
            "Попробуйте ввести другой (например, GOOGL):"
        )
        return TICKER 

    context.user_data['ticker'] = ticker
    
    await update.message.reply_text(
        f"✅ Тикер {ticker} найден!\n"
        "Введите **сумму инвестиций** в USD (например: 1000):",
        parse_mode='Markdown'
    )
    return AMOUNT

async def get_amount_and_process(update: Update, context: CallbackContext):
    """Главная функция, объединяющая все 5 этапов."""
    try:
        # Обработка ввода суммы
        amount_text = update.message.text.replace(',', '.')
        amount = float(amount_text)
        
        if amount <= 0:
            await update.message.reply_text("Сумма должна быть больше нуля.")
            return AMOUNT
            
        context.user_data['amount'] = amount
        ticker = context.user_data['ticker']
        user_id = update.message.from_user.id
        
        # --- ЭТАП 1: Загрузка данных ---
        await update.message.reply_text("📥 Загружаю данные за 2 года...")
        df = data_handler.load_data(ticker)
        
        if df.empty:
            await update.message.reply_text("❌ Ошибка загрузки данных. Попробуйте /start заново.")
            return ConversationHandler.END

        await update.message.reply_text(
            f"📂 Данные: {len(df)} строк.\n"
            "🤖 Обучаю модели (Random Forest, ARIMA, LSTM)...\n"
            "⏳ Ждите ~15 секунд..."
        )

        # --- ЭТАП 2: Обучение и выбор модели ---
        try:
            best_model_name, forecast, metrics = forecaster.train_and_predict(df)
        except Exception as e:
            logging.error(f"Ошибка ML: {e}")
            await update.message.reply_text("❌ Ошибка при обучении моделей.")
            return ConversationHandler.END

        # --- ЭТАП 3: Визуализация ---
        await update.message.reply_text(f"🏆 Победила модель: {best_model_name}")
        image_path = visualizer.plot_forecast(df, forecast, ticker)
        
        # Отправка фото
        if os.path.exists(image_path):
            with open(image_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo)
            os.remove(image_path) # Удаляем файл

        # Расчет процентов
        last_price = float(df['Close'].iloc[-1])
        last_forecast = float(forecast.iloc[-1])
        change_pct = ((last_forecast - last_price) / last_price) * 100
        emoji = "🚀" if change_pct > 0 else "🔻"

        # --- ЭТАП 4: Рекомендации ---
        recommendations, potential_profit = analysis.get_recommendations(forecast, amount)

        # --- ЭТАП 5: Логирование ---
        logger.log_request(user_id, ticker, amount, best_model_name, metrics, potential_profit)

        # Финальный отчет
        report = (
            f"📊 **Итоги для {ticker}**\n"
            f"🔹 Цена сейчас: ${last_price:.2f}\n"
            f"🔹 Прогноз (30 дн): ${last_forecast:.2f}\n"
            f"🔹 Тренд: {emoji} {change_pct:+.2f}%\n"
            f"🔹 Ошибка (RMSE): ${metrics['rmse']:.2f}\n"
            f"🔹 Точность (MAPE): {metrics['mape']:.2f}%\n\n"
            f"{recommendations}"
        )

        await update.message.reply_text(report, parse_mode='Markdown')
        await update.message.reply_text("🏁 Анализ завершен. Нажмите /start для нового запроса.")
        
        return ConversationHandler.END

    except ValueError:
        await update.message.reply_text("🔢 Введите число (например: 1000).")
        return AMOUNT
    except Exception as e:
        logging.error(f"Critical Error: {e}")
        await update.message.reply_text("Произошла ошибка сервера.")
        return ConversationHandler.END

async def cancel(update: Update, context: CallbackContext):
    await update.message.reply_text("⛔ Отменено. /start")
    return ConversationHandler.END

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_ticker)],
            AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_amount_and_process)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    application.add_handler(conv_handler)
    print("✅ Бот запущен! Нажмите Ctrl+C для остановки.")
    application.run_polling()

if __name__ == '__main__':
    main()