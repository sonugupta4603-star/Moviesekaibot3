import os
import logging
import threading
import base64
from flask import Flask, jsonify
import telebot
from openai import OpenAI

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("telegram_bot")

# ---------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# Model identifier requested for Hugging Face Router
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-V4.1-Flash:novita")

# Port for Render.com (Render assigns this automatically via $PORT)
PORT = int(os.environ.get("PORT", 5000))

if not BOT_TOKEN:
    logger.warning("WARNING: BOT_TOKEN is missing! Set it in your environment variables.")

if not HF_TOKEN:
    logger.warning("WARNING: HF_TOKEN is missing! Set it in your environment variables.")

# ---------------------------------------------------------
# Client Initialization
# ---------------------------------------------------------
# Connect to Hugging Face Router using OpenAI's client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN or "EMPTY",
)

# Initialize Telegram bot
bot = telebot.TeleBot(BOT_TOKEN or "EMPTY")

# Initialize Flask web server for Render health checks
app = Flask(__name__)


@app.route("/")
def index():
    return "Telegram Bot is running smoothly on Render!", 200


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "telegram-ai-bot"}), 200


# ---------------------------------------------------------
# Message Helper (Handles Telegram 4096 character limits)
# ---------------------------------------------------------
def send_safe_message(chat_id, text, reply_to_message_id=None):
    chunk_size = 4000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    for idx, chunk in enumerate(chunks):
        reply_id = reply_to_message_id if idx == 0 else None
        try:
            bot.send_message(chat_id, chunk, reply_to_message_id=reply_id, parse_mode="Markdown")
        except Exception:
            # Fallback to plain text if Markdown parsing fails
            try:
                bot.send_message(chat_id, chunk, reply_to_message_id=reply_id)
            except Exception as err:
                logger.error(f"Failed to deliver message chunk: {err}")


# ---------------------------------------------------------
# Telegram Bot Handlers
# ---------------------------------------------------------
@bot.message_handler(commands=["start"])
def handle_start(message):
    welcome_text = (
        "👋 *Hello! I am your AI Chatbot powered by Hugging Face.*\n\n"
        "✨ *Features:*\n"
        "• Send any text message to chat with me.\n"
        "• Send a photo (with an optional caption) to analyze or describe it.\n\n"
        "What would you like to ask?"
    )
    send_safe_message(message.chat.id, welcome_text, reply_to_message_id=message.message_id)


@bot.message_handler(commands=["help"])
def handle_help(message):
    help_text = (
        "📖 *Help & Instructions:*\n\n"
        "• Type your question or prompt to chat.\n"
        "• Send an image to inspect or describe it.\n\n"
        "Commands:\n"
        "/start - Restart or initialize the bot\n"
        "/help  - Show this help guide"
    )
    send_safe_message(message.chat.id, help_text, reply_to_message_id=message.message_id)


@bot.message_handler(content_types=["text"])
def handle_text_chat(message):
    if not HF_TOKEN:
        bot.reply_to(message, "⚠️ HF_TOKEN is not configured in the bot's environment variables.")
        return

    try:
        bot.send_chat_action(message.chat.id, "typing")

        # Chat completion via Hugging Face Router
        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful, respectful, and friendly AI chatbot.",
                },
                {
                    "role": "user",
                    "content": message.text,
                },
            ],
        )

        reply_content = chat_completion.choices[0].message.content
        if not reply_content:
            reply_content = "Received an empty response from the AI model."

        send_safe_message(message.chat.id, reply_content, reply_to_message_id=message.message_id)

    except Exception as e:
        logger.error(f"Error handling text message: {e}", exc_info=True)
        bot.reply_to(message, f"⚠️ An error occurred while generating a response:\n{str(e)}")


@bot.message_handler(content_types=["photo"])
def handle_photo_chat(message):
    if not HF_TOKEN:
        bot.reply_to(message, "⚠️ HF_TOKEN is not configured in the bot's environment variables.")
        return

    try:
        bot.send_chat_action(message.chat.id, "typing")

        # Get highest resolution image sent by user
        photo_obj = message.photo[-1]
        file_info = bot.get_file(photo_obj.file_id)
        image_bytes = bot.download_file(file_info.file_path)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt_text = message.caption if message.caption else "Describe this image in one sentence."

        # Vision completion via Hugging Face Router
        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )

        reply_content = chat_completion.choices[0].message.content
        if not reply_content:
            reply_content = "Received an empty response from the AI model."

        send_safe_message(message.chat.id, reply_content, reply_to_message_id=message.message_id)

    except Exception as e:
        logger.error(f"Error handling photo message: {e}", exc_info=True)
        bot.reply_to(message, f"⚠️ An error occurred while processing the image:\n{str(e)}")


# ---------------------------------------------------------
# Bot Polling Thread & Web Server Entry
# ---------------------------------------------------------
def start_bot_polling():
    logger.info("Starting Telegram bot polling thread...")
    try:
        bot.infinity_polling(skip_pending=True)
    except Exception as e:
        logger.error(f"Telegram bot polling exception: {e}", exc_info=True)


if __name__ == "__main__":
    if BOT_TOKEN:
        # Start Telegram bot in background thread
        bot_thread = threading.Thread(target=start_bot_polling, daemon=True)
        bot_thread.start()
    else:
        logger.error("BOT_TOKEN is missing! Set BOT_TOKEN in your environment variables.")

    # Start Flask web server for Render health checks
    logger.info(f"Starting Flask server on 0.0.0.0:{PORT}...")
    app.run(host="0.0.0.0", port=PORT)
