import requests
import time
from datetime import datetime

URL = "https://your-render-site.onrender.com"  # change this

INTERVAL = 300  # 5 minutes (in seconds)

while True:
    try:
        response = requests.get(URL, timeout=10)
        print(f"[{datetime.now()}] Status:", response.status_code)
    except Exception as e:
        print(f"[{datetime.now()}] Error:", e)

    time.sleep(INTERVAL)