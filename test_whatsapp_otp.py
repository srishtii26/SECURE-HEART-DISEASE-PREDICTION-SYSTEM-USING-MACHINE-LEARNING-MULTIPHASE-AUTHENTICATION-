from minimoth_otp import send_whatsapp_otp

PHONE_NUMBER = "+919606913791"

success, message, otp_id = send_whatsapp_otp(PHONE_NUMBER)

print("Success:", success)
print("Message:", message)
print("OTP ID:", otp_id)