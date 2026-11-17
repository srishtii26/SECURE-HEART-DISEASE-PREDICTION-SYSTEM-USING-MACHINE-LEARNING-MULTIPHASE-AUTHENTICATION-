import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MINIMOTH_API_KEY")

print("MiniMoth API key loaded:", bool(API_KEY))

if not API_KEY:
    raise ValueError("MINIMOTH_API_KEY not found in .env")

phone = input("Enter your WhatsApp number (+91XXXXXXXXXX): ").strip()

url = "https://api.minimoth.dev/v1/otp/send"

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json",
}

data = {
    "phone": phone
}

response = requests.post(
    url,
    headers=headers,
    json=data,
    timeout=30
)

print("\nStatus:", response.status_code)
print("Response:", response.text)