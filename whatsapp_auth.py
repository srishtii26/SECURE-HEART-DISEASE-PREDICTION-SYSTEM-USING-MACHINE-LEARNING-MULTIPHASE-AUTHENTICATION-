from login import login_user
from whatsapp_otp_security import (
    send_whatsapp_otp,
    verify_whatsapp_otp,
)


def authenticate_user(username, password):
    """
    Authenticate username/password first,
    then send and verify WhatsApp OTP.

    Returns:
        success, role, whatsapp_number, message
    """

    # Step 1: Verify username and password
    success, role, whatsapp_number = login_user(
        username,
        password
    )

    if not success:
        return False, None, None, "Invalid username or password."

    # Step 2: Check registered WhatsApp number
    if not whatsapp_number:
        return (
            False,
            None,
            None,
            "No WhatsApp number is registered for this user."
        )

    # Step 3: Send OTP
    otp_success, message, otp_id = send_whatsapp_otp(
        whatsapp_number
    )

    if not otp_success:
        return False, None, None, message

    return (
        True,
        role,
        whatsapp_number,
        otp_id,
    )


def verify_login_otp(phone_number, otp):
    """
    Verify the OTP sent through MiniMoth.
    """

    success, message, response_data = verify_whatsapp_otp(
        phone_number,
        otp
    )

    return success, message, response_data