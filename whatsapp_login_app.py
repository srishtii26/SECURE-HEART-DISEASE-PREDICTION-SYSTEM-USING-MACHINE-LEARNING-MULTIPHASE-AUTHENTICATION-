import streamlit as st

from login import login_user
from whatsapp_otp_security import (
    send_whatsapp_otp,
    verify_whatsapp_otp,
)


st.set_page_config(
    page_title="WhatsApp OTP Test",
    page_icon="🔐",
)


st.title("🔐 WhatsApp OTP Authentication Test")

# Initialize session state
if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False

if "otp_phone" not in st.session_state:
    st.session_state.otp_phone = None

if "username" not in st.session_state:
    st.session_state.username = None

if "role" not in st.session_state:
    st.session_state.role = None


# ============================================================
# LOGIN
# ============================================================

if not st.session_state.otp_sent:

    st.subheader("Step 1 — Login")

    username = st.text_input(
        "Username",
        placeholder="Enter username",
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password",
    )

    if st.button(
        "Login & Send WhatsApp OTP",
        type="primary",
        use_container_width=True,
    ):

        if not username or not password:
            st.warning("Please enter both username and password.")

        else:

            success, role, whatsapp_number = login_user(
                username,
                password,
            )

            if not success:
                st.error("Invalid username or password.")

            elif not whatsapp_number:
                st.error(
                    "No WhatsApp number is registered "
                    "for this account."
                )

            else:

                sent, message, otp_id = send_whatsapp_otp(
                    whatsapp_number
                )

                if sent:

                    st.session_state.otp_sent = True
                    st.session_state.otp_phone = whatsapp_number
                    st.session_state.username = username
                    st.session_state.role = role
                    st.session_state.otp_id = otp_id

                    st.success(
                        "OTP sent to your registered WhatsApp number!"
                    )

                    st.rerun()

                else:
                    st.error(message)


# ============================================================
# OTP VERIFICATION
# ============================================================

else:

    st.subheader("Step 2 — Verify WhatsApp OTP")

    st.info(
        "A 6-digit OTP has been sent to your registered "
        "WhatsApp number."
    )

    otp = st.text_input(
        "Enter OTP",
        max_chars=6,
        placeholder="Enter 6-digit OTP",
    )

    if st.button(
        "Verify OTP",
        type="primary",
        use_container_width=True,
    ):

        if not otp:
            st.warning("Please enter the OTP.")

        elif not otp.isdigit() or len(otp) != 6:
            st.error("OTP must be exactly 6 digits.")

        else:

            success, message, response = verify_whatsapp_otp(
                st.session_state.otp_phone,
                otp,
            )

            if success:

                st.success("✅ WhatsApp OTP verified successfully!")

                st.success(
                    f"🎉 Login successful!\n\n"
                    f"Username: {st.session_state.username}\n\n"
                    f"Role: {st.session_state.role}"
                )

                st.session_state.otp_sent = False

            else:
                st.error(message)

    if st.button("Cancel"):

        st.session_state.otp_sent = False
        st.session_state.otp_phone = None
        st.session_state.username = None
        st.session_state.role = None

        st.rerun()