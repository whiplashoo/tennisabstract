import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_URL = "https://www.tennisabstract.com"
REQUEST_DELAY = 1
REQUEST_RETRIES = 3

# For Telegram bot (optional, can be removed later)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
