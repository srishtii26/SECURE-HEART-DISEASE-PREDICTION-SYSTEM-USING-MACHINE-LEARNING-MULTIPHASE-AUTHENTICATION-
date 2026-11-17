import os
import requests
from dotenv import load_dotenv

load_dotenv()

MINIMOTH_API_KEY = os.getenv("MINIMOTH_API_KEY")

BASE_URL = "https://api.minimoth.dev/v1"


def send_whatsapp_otp(phone_number):
    """
    Request MiniMoth to generate and deliver an OTP.

    MiniMoth delivers through WhatsApp first and
    automatically falls back to SMS if necessary.

    Returns:
        success, message, otp_id
    """

    if not MINIMOTH_API_KEY:
        return False, "MiniMoth API key is not configured.", None

    headers = {
        "X-Api-Key": MINIMOTH_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "phone": phone_number
    }

    try:
        response = requests.post(
            f"{BASE_URL}/otp/send",
            headers=headers,
            json=payload,
            timeout=30,
        )

        data = response.json()

        if response.ok:
            return (
                True,
                "OTP sent successfully.",
                data.get("otp_id"),
            )

        return (
            False,
            data.get("error", response.text),
            None,
        )

    except requests.RequestException as e:
        return False, f"Network error: {e}", None

    except ValueError:
        return False, "Invalid response received from MiniMoth.", None


def verify_whatsapp_otp(phone_number, otp):
    """
    Verify the OTP entered by the user through MiniMoth.

    Returns:
        success, message, response_data
    """

    if not MINIMOTH_API_KEY:
        return False, "MiniMoth API key is not configured.", None

    headers = {
        "X-Api-Key": MINIMOTH_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "phone": phone_number,
        "code": otp,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/otp/verify",
            headers=headers,
            json=payload,
            timeout=30,
        )

        data = response.json()

        if response.ok:
            return (
                True,
                "WhatsApp OTP verified successfully.",
                data,
            )

        return (
            False,
            data.get("error", "Invalid OTP."),
            data,
        )

    except requests.RequestException as e:
        return False, f"Network error: {e}", None

    except ValueError:
        return False, "Invalid response received from MiniMoth.", None


def check_delivery_status(otp_id):
    """
    Check whether MiniMoth delivered the OTP through
    WhatsApp or SMS.
    """

    if not MINIMOTH_API_KEY:
        return False, "MiniMoth API key is not configured.", None

    headers = {
        "X-Api-Key": MINIMOTH_API_KEY,
    }

    try:
        response = requests.get(
            f"{BASE_URL}/otp/status/{otp_id}",
            headers=headers,
            timeout=30,
        )

        data = response.json()

        if response.ok:
            return True, "Delivery status retrieved.", data

        return False, data.get("error", response.text), data

    except requests.RequestException as e:
        return False, f"Network error: {e}", None

    except ValueError:
        return False, "Invalid response received from MiniMoth.", None