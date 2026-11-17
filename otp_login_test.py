from login import login_user
from whatsapp_otp_security import (
    send_whatsapp_otp,
    verify_whatsapp_otp,
)


def whatsapp_login(username, password):
    """
    Test the complete login flow:

    1. Check username/password.
    2. Get the user's registered WhatsApp number.
    3. Send OTP through MiniMoth.
    4. Ask the user for the OTP.
    5. Verify the OTP.
    """

    # Step 1: Check username and password
    success, role, whatsapp_number = login_user(username, password)

    if not success:
        print("Invalid username or password.")
        return False

    # Step 2: Make sure a WhatsApp number exists
    if not whatsapp_number:
        print("No WhatsApp number is registered for this user.")
        return False

    print(f"Password verified for {username}.")
    print("Sending OTP to registered WhatsApp number...")

    # Step 3: Send OTP
    sent, message, otp_id = send_whatsapp_otp(whatsapp_number)

    if not sent:
        print("Could not send OTP.")
        print("Reason:", message)
        return False

    print("OTP sent successfully!")
    print("OTP ID:", otp_id)

    # Step 4: Ask user to enter OTP
    entered_otp = input("Enter the 6-digit OTP received on WhatsApp: ").strip()

    # Step 5: Verify OTP
    verified, verify_message, response = verify_whatsapp_otp(
        whatsapp_number,
        entered_otp,
    )

    if verified:
        print("\nLogin successful!")
        print("Username:", username)
        print("Role:", role)
        return True

    print("\nOTP verification failed.")
    print("Reason:", verify_message)
    return False


if __name__ == "__main__":

    username = input("Username: ").strip()
    password = input("Password: ")

    whatsapp_login(username, password)