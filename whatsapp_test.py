import os

from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_number = os.getenv("TWILIO_WHATSAPP_FROM")

print("Account SID loaded:", bool(account_sid))
print("Auth Token loaded:", bool(auth_token))
print("WhatsApp sender:", from_number)

to_number = input("Enter your WhatsApp number (+91XXXXXXXXXX): ").strip()

client = Client(account_sid, auth_token)

try:
    message = client.messages.create(
        from_=from_number,
        to=f"whatsapp:{to_number}",
        body="SecureHeartDisease test message. WhatsApp connection test."
    )

    print()
    print("SUCCESS!")
    print("Message SID:", message.sid)

except TwilioRestException as e:
    print()
    print("TWILIO ERROR")
    print("Status:", e.status)
    print("Code:", e.code)
    print("Message:", e.msg)