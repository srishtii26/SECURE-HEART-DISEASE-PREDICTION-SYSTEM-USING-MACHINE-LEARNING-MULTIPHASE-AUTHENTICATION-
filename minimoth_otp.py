import os
import requests
from dotenv import load_dotenv

load_dotenv()

MINIMOTH_API_KEY = os.getenv("MINIMOTH_API_KEY")

API_URL = "https://api.minimoth.dev/v1/otp/send"


def send_whatsapp_otp(phone_number):
    """Send an OTP to the user's phone through MiniMoth."""

    if not MINIMOTH_API_KEY:
        return False, "MiniMoth API key is not configured.", None

    headers = {
        "x-api-key": MINIMOTH_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "phone": phone_number
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        if response.ok:
            data = response.json()
            otp_id = data.get("otp_id")

            return True, "OTP sent successfully.", otp_id

        return False, f"MiniMoth error {response.status_code}: {response.text}", None

    except requests.RequestException as e:
        return False, f"Network error: {e}", None