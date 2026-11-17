import streamlit as st
import pandas as pd
import time
import sqlite3
from datetime import datetime 
from ecg.ecg_ui import show_ecg_analysis
from otp_security import create_otp, verify_otp
from whatsapp_auth import (
    authenticate_user,
    verify_login_otp,
)
from minimoth_otp import send_whatsapp_otp

from login import login_user, register_user
from predict import predict_heart_disease, load_model
from utils import (
    validate_patient_data,
    log_prediction,
    verify_model_integrity,
    TRUSTED_MODEL_HASH,
)
from database import get_prediction_logs, get_login_logs
from security.adversarial_defense import (
    calculate_feature_statistics,
    detect_input_anomaly,
    check_prediction_stability,
)
from security.extraction_defense import check_extraction_risk
from admin_dashboard import show_admin_dashboard
from matrix_security import (
    generate_matrix,
    generate_challenge,
    verify_matrix,
)
from face_auth_streamlit import train_recognizer, verify_camera_image
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Secure Heart",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/heart_model.pkl"
DATASET_PATH = "dataset/heart.csv"


# ============================================================
# PREMIUM UI / UX
# ============================================================

st.markdown(
    """
<style>
/* ---------- GLOBAL ---------- */
:root {
    --bg: #08090d;
    --bg2: #0d1016;
    --panel: #11141b;
    --panel2: #151922;
    --border: rgba(255,255,255,.075);
    --border2: rgba(255,255,255,.12);
    --text: #f6f7fb;
    --muted: #9299a8;
    --accent: #ff4d5a;
    --accent2: #ff8a5b;
    --green: #38d996;
    --yellow: #f4c451;
    --blue: #65a8ff;
    --danger: #ff5b68;
}

html, body,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 90% 0%, rgba(255,77,90,.055), transparent 25%),
        radial-gradient(circle at 10% 100%, rgba(101,168,255,.035), transparent 25%),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(8,9,13,.78);
}

[data-testid="stMain"] {
    background: transparent;
}

.block-container {
    max-width: 1240px;
    padding-top: 2.2rem;
    padding-bottom: 5rem;
}

h1, h2, h3, h4 {
    letter-spacing: -0.035em;
    color: var(--text);
}

p, label, .stMarkdown {
    color: var(--text);
}

small, .muted {
    color: var(--muted);
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #0c0f15 0%, #090b10 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 4px 20px 4px;
}

.sidebar-mark {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    box-shadow: 0 12px 34px rgba(255,77,90,.22);
    font-size: 21px;
}

.sidebar-name {
    font-size: 19px;
    font-weight: 800;
    line-height: 1;
}

.sidebar-sub {
    margin-top: 5px;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: .12em;
    text-transform: uppercase;
}

.user-card {
    padding: 15px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,.025);
    border-radius: 16px;
    margin-bottom: 18px;
}

.user-name {
    font-weight: 700;
    font-size: 15px;
}

.user-role {
    margin-top: 4px;
    color: var(--muted);
    font-size: 12px;
}

.nav-caption {
    color: var(--muted);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .15em;
    text-transform: uppercase;
    margin: 18px 4px 8px;
}

/* ---------- HERO ---------- */
.hero {
    position: relative;
    overflow: hidden;
    min-height: 280px;
    padding: 46px 48px;
    border-radius: 28px;
    border: 1px solid var(--border);
    background:
        radial-gradient(circle at 87% 15%, rgba(255,77,90,.24), transparent 31%),
        radial-gradient(circle at 12% 100%, rgba(101,168,255,.10), transparent 32%),
        linear-gradient(135deg, #161a23 0%, #0e1118 72%);
    box-shadow: 0 30px 80px rgba(0,0,0,.22);
    margin-bottom: 24px;
}

.hero:after {
    content: "♥";
    position: absolute;
    right: 7%;
    bottom: -35px;
    font-size: 210px;
    color: rgba(255,255,255,.025);
    font-weight: 900;
    line-height: 1;
}

.hero-kicker {
    color: #ff8a93;
    text-transform: uppercase;
    letter-spacing: .18em;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 13px;
}

.hero-title {
    max-width: 720px;
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1.02;
    font-weight: 850;
    letter-spacing: -.055em;
    margin-bottom: 16px;
}

.hero-title span {
    background: linear-gradient(90deg, #ffffff 25%, #ff9a91 85%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-text {
    max-width: 650px;
    color: #a9afbc;
    font-size: 15px;
    line-height: 1.75;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-top: 22px;
    padding: 8px 12px;
    border: 1px solid rgba(56,217,150,.18);
    background: rgba(56,217,150,.065);
    border-radius: 999px;
    color: #77e5b3;
    font-size: 11px;
    font-weight: 700;
}

/* ---------- SECTION HEADERS ---------- */
.section-kicker {
    color: #ff7881;
    text-transform: uppercase;
    letter-spacing: .16em;
    font-size: 10px;
    font-weight: 800;
    margin-bottom: 7px;
}

.section-title {
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 20px;
}

/* ---------- CARDS ---------- */
.card {
    padding: 22px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: linear-gradient(145deg, rgba(255,255,255,.035), rgba(255,255,255,.018));
    box-shadow: 0 18px 45px rgba(0,0,0,.12);
}

.card-title {
    font-size: 15px;
    font-weight: 750;
    margin-bottom: 5px;
}

.card-subtitle {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.55;
}

/* ---------- SECURITY STATUS ---------- */
.security-card {
    min-height: 142px;
    padding: 19px;
    border-radius: 18px;
    border: 1px solid var(--border);
    background: #11151d;
    transition: transform .18s ease, border-color .18s ease;
}

.security-card:hover {
    transform: translateY(-2px);
    border-color: var(--border2);
}

.security-card.good {
    box-shadow: inset 3px 0 0 var(--green);
}

.security-card.warn {
    box-shadow: inset 3px 0 0 var(--yellow);
}

.security-card.bad {
    box-shadow: inset 3px 0 0 var(--danger);
}

.security-icon {
    font-size: 21px;
    margin-bottom: 10px;
}

.security-name {
    font-size: 14px;
    font-weight: 750;
}

.security-status {
    margin-top: 8px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .08em;
}

.good .security-status { color: var(--green); }
.warn .security-status { color: var(--yellow); }
.bad .security-status { color: var(--danger); }

/* ---------- FORM ---------- */
.form-shell {
    padding: 25px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: linear-gradient(145deg, #11151d, #0d1016);
    margin-top: 8px;
}

.form-group-title {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: .02em;
    margin: 5px 0 16px;
    color: #e9ebf1;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #aeb4c1 !important;
    font-size: 11px !important;
    font-weight: 650 !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] input {
    background: #171a22 !important;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
    border-radius: 13px !important;
    min-height: 44px;
    font-weight: 750 !important;
    border: 1px solid var(--border2) !important;
    background: #171a22 !important;
    color: #f7f8fb !important;
    transition: all .18s ease !important;
}

.stButton > button:hover {
    border-color: rgba(255,77,90,.5) !important;
    transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #ff4d5a, #ff765b) !important;
    border: none !important;
    box-shadow: 0 12px 32px rgba(255,77,90,.18) !important;
}
/* ---------- REGISTER ---------- */

.register-header {
    text-align: center;
    margin: 35px auto 25px;
}

.register-icon {
    width: 58px;
    height: 58px;
    margin: 0 auto 16px;
    border-radius: 17px;
    display: grid;
    place-items: center;
    font-size: 28px;
    background: linear-gradient(
        135deg,
        #ff4d5a,
        #ff765b
    );
    box-shadow: 0 18px 45px rgba(255,77,90,.24);
}

.register-title {
    font-size: 34px;
    font-weight: 850;
    letter-spacing: -.05em;
    color: #f7f8fb;
}

.register-subtitle {
    color: #8f96a5;
    font-size: 13px;
    margin-top: 7px;
}

.register-section-title {
    color: #e9ebf1;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 5px;
}

.register-section-subtitle {
    color: #7f8796;
    font-size: 11px;
    line-height: 1.5;
    margin-bottom: 16px;
}

.register-security-note {
    text-align: center;
    color: #7f8796;
    font-size: 10px;
    margin: 15px 0 12px;
    line-height: 1.7;
}

/* ---------- LOGIN ---------- */
.login-page {
    min-height: 78vh;
    display: flex;
    align-items: center;
    justify-content: center;
}

.login-card {
    max-width: 470px;
    margin: 0 auto;
    padding: 38px;
    border: 1px solid var(--border);
    border-radius: 26px;
    background:
        radial-gradient(circle at 100% 0%, rgba(255,77,90,.14), transparent 34%),
        linear-gradient(145deg, #141821, #0d1016);
    box-shadow: 0 30px 90px rgba(0,0,0,.30);
}

.login-logo {
    width: 58px;
    height: 58px;
    border-radius: 17px;
    display: grid;
    place-items: center;
    font-size: 28px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    box-shadow: 0 18px 45px rgba(255,77,90,.24);
    margin-bottom: 20px;
}

.login-title {
    font-size: 34px;
    font-weight: 850;
    letter-spacing: -.05em;
}

.login-subtitle {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
    margin: 8px 0 25px;
}

/* ---------- RESULT ---------- */
.result-card {
    padding: 30px;
    border-radius: 23px;
    border: 1px solid var(--border);
    margin-top: 12px;
    background: linear-gradient(145deg, #131720, #0e1117);
}

.result-card.safe {
    box-shadow: inset 4px 0 0 var(--green);
}

.result-card.risk {
    box-shadow: inset 4px 0 0 var(--danger);
}

.result-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .16em;
}

.result-title {
    font-size: 28px;
    font-weight: 850;
    margin: 8px 0 15px;
}

.confidence-value {
    font-size: 42px;
    font-weight: 850;
    letter-spacing: -.04em;
}

.confidence-label {
    color: var(--muted);
    font-size: 11px;
}

/* ---------- SCORE ---------- */
.score-card {
    padding: 23px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: #11151d;
    text-align: center;
}

.score-number {
    font-size: 48px;
    font-weight: 850;
    letter-spacing: -.05em;
}

.score-caption {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .13em;
}

/* ---------- METRICS ---------- */
.metric-card {
    padding: 20px;
    min-height: 112px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: #11151d;
}

.metric-label {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.metric-value {
    margin-top: 7px;
    font-size: 30px;
    font-weight: 850;
}

/* ---------- TABLE ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 15px;
    overflow: hidden;
}

/* ---------- ALERTS ---------- */
div[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid var(--border);
}

/* ---------- DIVIDER ---------- */
hr {
    border-color: var(--border) !important;
}

/* ---------- MOBILE ---------- */
@media (max-width: 800px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 30px 25px;
        min-height: 250px;
    }

    .hero-title {
        font-size: 38px;
    }

    .login-card {
        padding: 28px 22px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "logged_in": False,
    "username": None,
    "role": None,
    "last_activity": None,
    "session_created": None,
    "matrix_pending": False,
    "matrix_value": None,
    "matrix_positions": None,
    "matrix_attempts": 0,
    "otp_pending": False,
    "otp_record": None,
    "otp_username": None,
    "otp_role": None,
    "otp_phone": None,
    "generated_otp": None,
    "otp_whatsapp": None,
    "prediction_result": None,
    "confidence": None,
    "anomaly_result": None,
    "stability_result": None,
    "page": "Prediction",
    "show_register": False,
    "face_pending": False,
    "face_attempts": 0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
        # Session timeout: 15 minutes
SESSION_TIMEOUT = 2*60


# ============================================================
# DATA / SECURITY HELPERS
# ============================================================

@st.cache_resource
def load_feature_statistics():
    training_df = pd.read_csv(DATASET_PATH)
    return calculate_feature_statistics(training_df)


def model_integrity_ok():
    try:
        return verify_model_integrity(
            MODEL_PATH,
            TRUSTED_MODEL_HASH,
        )
    except Exception:
        return False


def reset_prediction():
    st.session_state.prediction_result = None
    st.session_state.confidence = None
    st.session_state.anomaly_result = None
    st.session_state.stability_result = None


def logout(auto=False):
    username = st.session_state.get("username")

    # Log automatic logout separately
    if auto and username:
        try:
            from login import log_login
            log_login(username, "AUTO_LOGOUT")
        except Exception:
            pass

    reset_prediction()

    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.face_pending = False
    st.session_state.face_attempts = 0
    st.session_state.last_activity = None
    st.session_state.session_created = None
    st.session_state.page = "Prediction"

    # Clear OTP/session information
    st.session_state.otp_pending = False
    st.session_state.otp_record = None
    st.session_state.otp_username = None
    st.session_state.otp_role = None
    st.session_state.otp_phone = None
    st.session_state.generated_otp = None
    st.session_state.otp_whatsapp = None

    st.rerun()


def check_session_timeout():
    """
    Automatically logs out the user after 15 minutes of inactivity.
    """

    if not st.session_state.get("logged_in"):
        return True

    current_time = time.time()
    last_activity = st.session_state.get("last_activity")

    # Start the timer if it doesn't exist
    if last_activity is None:
        st.session_state.last_activity = current_time
        st.session_state.session_created = current_time
        return True

    # Check whether the session has expired
    if current_time - last_activity > SESSION_TIMEOUT:
        st.warning(
            "Your session has expired due to inactivity. "
            "Please log in again."
        )
        logout(auto=True)
        return False

    # User is active, so update the activity time
    st.session_state.last_activity = current_time

    return True


# ============================================================
# BRAND
# ============================================================

def show_brand(compact=False):
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-mark">♥</div>
            <div>
                <div class="sidebar-name">SECURE HEART</div>
                <div class="sidebar-sub">AI • HEALTH • SECURITY</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_otp_verification():
    st.markdown("### 🔐 Two-Factor Authentication")

    st.info(
        "Your username and password were verified. "
        "A 6-digit OTP has been sent to your registered WhatsApp number."
    )

    otp_input = st.text_input(
        "Enter WhatsApp OTP",
        max_chars=6,
        placeholder="Enter 6-digit OTP",
        key="otp_input",
    )

    col1, col2 = st.columns(2)

    # ---------------- VERIFY OTP ----------------
    with col1:
        if st.button(
            "Verify OTP",
            type="primary",
            use_container_width=True,
        ):
            if not otp_input:
                st.warning("Please enter the OTP.")
                return

            if not otp_input.isdigit() or len(otp_input) != 6:
                st.warning("OTP must be exactly 6 digits.")
                return

            success, message, response_data = verify_login_otp(
                st.session_state.otp_phone,
                otp_input.strip(),
            )

            if success:
                # OTP verified successfully
                st.session_state.otp_pending = False
                st.session_state.otp_phone = None
                st.session_state.otp_id = None

                # Prepare Matrix Verification
                st.session_state.matrix_pending = True
                st.session_state.matrix_positions = generate_challenge()
                st.session_state.matrix_attempts = 0
                st.session_state.matrix_answer = ""

                st.success("WhatsApp OTP verified successfully!")

                st.rerun()

            else:
                st.error(message)

    # ---------------- RESEND OTP ----------------
    with col2:
        if st.button(
            "🔄 Resend OTP",
            use_container_width=True,
        ):
            from whatsapp_otp_security import send_whatsapp_otp

            phone_number = st.session_state.otp_phone

            if not phone_number:
                st.error("No WhatsApp number is associated with this login.")
                return

            success, message, otp_id = send_whatsapp_otp(
                phone_number
            )

            if success:
                st.session_state.otp_id = otp_id
                st.success("A new OTP has been sent to WhatsApp.")
                st.rerun()
            else:
                st.error(message)

    # ---------------- CANCEL ----------------
    if st.button(
        "Cancel Login",
        use_container_width=True,
    ):
        st.session_state.otp_pending = False
        st.session_state.otp_phone = None
        st.session_state.otp_id = None
        st.session_state.otp_username = None
        st.session_state.otp_role = None
        st.rerun()


def show_matrix_verification():
    st.markdown("### 🔐 Matrix Verification")

    username = st.session_state.get("otp_username")

    if not username:
        st.error(
            "Matrix verification session expired. "
            "Please log in again."
        )
        return
    @st.cache_resource
    def load_admin_face_model():
        return train_recognizer()

@st.cache_resource
def load_admin_face_model():
    return train_recognizer()


def show_matrix_verification():

    st.markdown("### 🔐 Matrix Verification")

    username = st.session_state.get("otp_username")

    if not username:
        st.error(
            "Matrix verification session expired. "
            "Please log in again."
        )
        return

    # Load the user's permanent 3x4 matrix
    if st.session_state.get("matrix_value") is None:
        st.session_state.matrix_value = generate_matrix(username)

    matrix = st.session_state.matrix_value

    # Generate a new challenge
    if st.session_state.get("matrix_positions") is None:
        st.session_state.matrix_positions = generate_challenge()

    positions = st.session_state.matrix_positions

    labels = [
        f"R{row + 1}C{col + 1}"
        for row, col in positions
    ]

    st.info(
        "Use your permanent security matrix/card and enter "
        "the characters from the requested positions in the same order."
    )

    st.markdown("#### 🔑 Matrix Challenge")

    st.markdown(
        "**Enter the characters at these positions "
        "in the same order:**"
    )

    st.markdown(
        " → ".join(f"`{label}`" for label in labels)
    )

    st.markdown("---")

    answer = st.text_input(
        "Matrix Code",
        max_chars=3,
        placeholder="Enter 3 characters",
        key="matrix_answer",
    )

    st.caption(
        "The permanent matrix is not displayed here. "
        "Use your assigned physical matrix/card."
    )

    if st.button(
        "Verify Matrix",
        type="primary",
        use_container_width=True,
    ):

        answer = answer.strip()

        if len(answer) != 3:
            st.warning("Please enter exactly 3 characters.")
            return

        success = verify_matrix(
            matrix,
            positions,
            answer,
        )

        if success:

            st.session_state.matrix_pending = False
            st.session_state.matrix_value = None
            st.session_state.matrix_positions = None
            st.session_state.matrix_attempts = 0

            if st.session_state.get("otp_role") == "admin":

                st.session_state.face_pending = True
                st.session_state.face_attempts = 0

            else:

                st.session_state.logged_in = True
                st.session_state.last_activity = time.time()
                st.session_state.session_created = time.time()

                st.session_state.username = st.session_state.otp_username
                st.session_state.role = st.session_state.otp_role
                st.session_state.page = "Prediction"

                st.session_state.otp_username = None
                st.session_state.otp_role = None

            st.success("Matrix verification successful!")
            st.rerun()

        else:

            st.session_state.matrix_attempts += 1

            if st.session_state.matrix_attempts >= 3:

                st.error(
                    "Too many incorrect matrix attempts. "
                    "Please log in again."
                )

                st.session_state.matrix_pending = False
                st.session_state.matrix_value = None
                st.session_state.matrix_positions = None
                st.session_state.matrix_attempts = 0
                st.session_state.otp_username = None
                st.session_state.otp_role = None

                st.rerun()

            else:

                remaining = 3 - st.session_state.matrix_attempts

                st.error("Incorrect matrix response.")
                st.warning(f"Attempts remaining: {remaining}")


def show_face_verification():

    st.title("Admin Face Verification")

    st.info(
        "Please position your face clearly in front of the camera "
        "to complete Admin authentication."
    )

    if st.button("Cancel and Logout"):
        logout()
        st.rerun()

    try:
        recognizer, names, detector = load_admin_face_model()

    except Exception as e:
        st.error(f"Face recognition system error: {e}")
        st.warning(
            "Admin login cannot continue without face verification."
        )
        return

    captured_image = st.camera_input(
        "Capture your face",
        key="admin_face_camera"
    )

    if captured_image is not None:

        if st.button("Verify Face", use_container_width=True):

            matched_name, distance, preview = verify_camera_image(
                captured_image.getvalue(),
                recognizer,
                names,
                detector
            )

            if matched_name is not None:

                st.success(f"Face recognized: {matched_name}")

                st.session_state.face_pending = False
                st.session_state.face_attempts = 0

                st.session_state.logged_in = True
                st.session_state.username = st.session_state.otp_username
                st.session_state.role = "admin"

                st.session_state.last_activity = time.time()
                st.session_state.session_created = time.time()
                st.session_state.page = "Prediction"

                st.session_state.otp_pending = False
                st.session_state.otp_username = None
                st.session_state.otp_role = None

                st.success("Admin authentication successful!")
                st.rerun()

            else:

                st.session_state.face_attempts += 1

                st.error("Face not recognized. Please try again.")

                st.write(
                    f"Failed attempts: "
                    f"{st.session_state.face_attempts}"
                )

# ============================================================
# REGISTER PAGE
# ============================================================

def register_page():

    # --------------------------------------------------------
    # REGISTER HEADER
    # --------------------------------------------------------

    st.markdown(
        """<div style="max-width:540px;margin:35px auto 20px;text-align:center;">
<div style="width:58px;height:58px;margin:0 auto 15px;border-radius:17px;display:flex;align-items:center;justify-content:center;font-size:28px;background:linear-gradient(135deg,#ff4d5a,#ff765b);box-shadow:0 18px 45px rgba(255,77,90,.24);">♥</div>
<div style="font-size:32px;font-weight:850;color:#f7f8fb;">Create your account</div>
<div style="color:#8f96a5;font-size:13px;margin-top:7px;">Create a secure account for Secure Heart.</div>
</div>""",
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.25, 1])

    with center:

        # ----------------------------------------------------
        # ACCOUNT INFORMATION
        # ----------------------------------------------------

        st.markdown(
            """<div style="margin-bottom:18px;">
<div style="color:#e9ebf1;font-size:14px;font-weight:800;margin-bottom:5px;">Account Information</div>
<div style="color:#7f8796;font-size:11px;line-height:1.5;">Create your Secure Heart user account.</div>
</div>""",
            unsafe_allow_html=True,
        )

        username = st.text_input(
            "Username",
            placeholder="Choose a unique username",
            key="register_username",
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Create a password",
            key="register_password",
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password",
            key="register_confirm_password",
        )

        # ----------------------------------------------------
        # WHATSAPP
        # ----------------------------------------------------

        st.markdown(
            """<div style="margin-top:20px;margin-bottom:15px;">
<div style="color:#e9ebf1;font-size:14px;font-weight:800;margin-bottom:5px;">Recovery & Verification</div>
<div style="color:#7f8796;font-size:11px;line-height:1.5;">Your WhatsApp number is used for two-factor authentication.</div>
</div>""",
            unsafe_allow_html=True,
        )

        whatsapp = st.text_input(
            "WhatsApp Number",
            placeholder="+91XXXXXXXXXX",
            key="register_whatsapp",
        )

        st.markdown(
            """<div style="text-align:center;color:#727989;font-size:10px;margin:15px 0;line-height:1.7;">
🔐 Password protected &nbsp; • &nbsp;
📱 WhatsApp verification &nbsp; • &nbsp;
🔑 Permanent matrix authentication
</div>""",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # CREATE ACCOUNT
        # ----------------------------------------------------

        if st.button(
            "Create Account  →",
            type="primary",
            use_container_width=True,
            key="create_account_button",
        ):

            username = username.strip()
            whatsapp = whatsapp.strip()

            # Validation
            if not username:

                st.error(
                    "Username is required."
                )

            elif len(username) < 4:

                st.error(
                    "Username must contain at least 4 characters."
                )

            elif not password:

                st.error(
                    "Password is required."
                )

            elif len(password) < 6:

                st.error(
                    "Password must contain at least 6 characters."
                )

            elif password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            elif not whatsapp:

                st.error(
                    "WhatsApp number is required."
                )

            else:

                # Create user using existing login.py
                success, message = register_user(
                    username,
                    password,
                    "user",
                    whatsapp,
                )

                if success:

                    # Create permanent security matrix
                    generate_matrix(username)

                    st.success(
                        "Account created successfully!"
                    )

                    st.info(
                        "Your permanent security matrix has "
                        "also been created. Please contact the "
                        "administrator to receive your physical card."
                    )

                    st.session_state.show_register = False

                    # Clear registration fields
                    for key in [
                        "register_username",
                        "register_password",
                        "register_confirm_password",
                        "register_whatsapp",
                    ]:
                        st.session_state.pop(key, None)

                    st.rerun()

                else:

                    st.error(message)

        # ----------------------------------------------------
        # BACK TO LOGIN
        # ----------------------------------------------------

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        if st.button(
            "← Back to Login",
            use_container_width=True,
            key="back_to_login_button",
        ):

            st.session_state.show_register = False

            for key in [
                "register_username",
                "register_password",
                "register_confirm_password",
                "register_whatsapp",
            ]:
                st.session_state.pop(key, None)

            st.rerun()


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    # --------------------------------------------------------
    # REGISTER PAGE
    # --------------------------------------------------------

    if st.session_state.get("show_register", False):

        register_page()

        return

    # --------------------------------------------------------
    # OTP VERIFICATION
    # --------------------------------------------------------

    if st.session_state.get("otp_pending", False):

        show_otp_verification()

        return

    # --------------------------------------------------------
    # LOGIN PAGE CONTAINER
    # --------------------------------------------------------

    st.markdown(
        '<div class="login-page">',
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1, 1.25, 1])

    with center:

        # ====================================================
        # LOGIN CARD
        # ====================================================

        st.markdown(
            """<div class="login-card"><div class="login-logo">♥</div><div class="login-title">Welcome back.</div><div class="login-subtitle">Secure access to an AI-powered heart disease prediction platform with built-in model and input protection.</div></div>""",
            unsafe_allow_html=True,
        )

        # ====================================================
        # EMAIL / NAME
        # ====================================================

        login_id = st.text_input(
            "Email or Name",
            placeholder="Enter your email or name",
            key="login_username",
        )

        # ====================================================
        # PASSWORD
        # ====================================================

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password",
        )

        # ====================================================
        # LOGIN BUTTON
        # ====================================================

        if st.button(
            "Continue to Secure Heart  →",
            type="primary",
            use_container_width=True,
            key="login_button",
        ):

            if not login_id or not password:

                st.warning(
                    "Please enter your email/name and password."
                )

            else:

                # ------------------------------------------------
                # FRIENDLY NAME / EMAIL → EXISTING USERNAME
                # ------------------------------------------------
                #
                # The internal usernames remain user1, user2,
                # etc. This keeps their existing permanent
                # matrix files unchanged.
                #

                account_map = {

                    # USER 1
                    "aarav mehta": "user1",
                    "aarav.mehta@gmail.com": "user1",

                    # USER 2
                    "rahul sharma": "user2",
                    "rahul.sharma@gmail.com": "user2",

                    # USER 3
                    "ananya iyer": "user3",
                    "ananya.iyer@gmail.com": "user3",

                    # USER 4
                    "rohan kumar": "user4",
                    "rohan.kumar@gmail.com": "user4",

                    # USER 5
                    "meera nair": "user5",
                    "meera.nair@gmail.com": "user5",

                    # USER 6
                    "arjun menon": "user6",
                    "arjun.menon@gmail.com": "user6",

                    # USER 7
                    "kavya reddy": "user7",
                    "kavya.reddy@gmail.com": "user7",

                    # ADMIN
                    "system administrator": "admin",
                    "admin@secureheart.com": "admin",
                    "admin": "admin",

                    # TEST ACCOUNT
                    "test user": "testuser123",
                    "testuser@secureheart.com": "testuser123",
                }

                entered_login = login_id.strip().lower()

                internal_username = account_map.get(
                    entered_login,
                    login_id.strip()
                )

                # ------------------------------------------------
                # EXISTING AUTHENTICATION SYSTEM
                # ------------------------------------------------

                try:

                    success, role, whatsapp_number, result = (
                        authenticate_user(
                            internal_username,
                            password,
                        )
                    )

                except Exception as error:

                    success = False
                    role = None
                    whatsapp_number = None

                    result = (
                        "Authentication service error: "
                        + str(error)
                    )

                # ------------------------------------------------
                # SUCCESSFUL LOGIN
                # ------------------------------------------------

                if success:

                    # --------------------------------------------
                    # STORE OTP STATE
                    # --------------------------------------------

                    st.session_state.otp_pending = True

                    # IMPORTANT:
                    # Keep the INTERNAL username here.
                    # This preserves the existing matrix.
                    st.session_state.otp_username = (
                        internal_username
                    )

                    st.session_state.otp_role = role

                    st.session_state.otp_phone = (
                        whatsapp_number
                    )

                    st.session_state.otp_id = result

                    st.success(
                        "OTP sent to your WhatsApp number."
                    )

                    st.rerun()

                # ------------------------------------------------
                # FAILED LOGIN
                # ------------------------------------------------

                else:

                    st.error(result)

        # ====================================================
        # SECURITY FOOTER
        # ====================================================

        st.markdown(
            """<div style="text-align:center;color:#727989;font-size:11px;margin-top:18px;line-height:1.6;">Protected authentication • Activity logging enabled</div>""",
            unsafe_allow_html=True,
        )

        # ====================================================
        # CREATE ACCOUNT SECTION
        # ====================================================

        st.markdown(
            """<div style="text-align:center;color:#727989;font-size:12px;margin-top:20px;margin-bottom:8px;">Don't have an account?</div>""",
            unsafe_allow_html=True,
        )

        if st.button(
            "Create Account",
            use_container_width=True,
            key="open_register_button",
        ):

            st.session_state.show_register = True

            st.rerun()

    # --------------------------------------------------------
    # CLOSE LOGIN PAGE CONTAINER
    # --------------------------------------------------------

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

def show_sidebar():

    with st.sidebar:

        # ----------------------------------------------------
        # BRAND
        # ----------------------------------------------------

        show_brand()

        # ----------------------------------------------------
        # CURRENT USER
        # ----------------------------------------------------

        st.markdown(
            f"""<div class="user-card"><div class="user-name">👤 {st.session_state.username}</div><div class="user-role">{str(st.session_state.role).upper()} ACCESS</div></div>""",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # WORKSPACE
        # ----------------------------------------------------

        st.markdown(
            '<div class="nav-caption">Workspace</div>',
            unsafe_allow_html=True,
        )

        pages = [
            "Prediction",
            "Security Center",
        ]

        # ----------------------------------------------------
        # ADMIN-ONLY PAGES
        # ----------------------------------------------------

        if st.session_state.role == "admin":

            pages.append(
                "Admin Dashboard"
            )

            pages.append(
                "Matrix Management"
            )

        current = st.session_state.page

        if current not in pages:

            current = pages[0]

        selected = st.radio(
            "Navigation",
            pages,
            index=pages.index(current),
            label_visibility="collapsed",
        )

        st.session_state.page = selected

        # ----------------------------------------------------
        # SYSTEM
        # ----------------------------------------------------

        st.markdown(
            '<div class="nav-caption">System</div>',
            unsafe_allow_html=True,
        )

        integrity = model_integrity_ok()

        if integrity:

            status_text = "MODEL VERIFIED"
            status_class = "good"

        else:

            status_text = "MODEL BLOCKED"
            status_class = "bad"

        st.markdown(
            f"""<div class="security-card {status_class}" style="min-height:auto;"><div class="security-icon">🔐</div><div class="security-name">Model Integrity</div><div class="security-status">{status_text}</div></div>""",
            unsafe_allow_html=True,
        )

        st.write("")

        # ----------------------------------------------------
        # LOGOUT
        # ----------------------------------------------------

        if st.button(
            "Logout",
            use_container_width=True,
        ):

            logout()
# ============================================================
# SECURITY CARD
# ============================================================

def security_card(icon, title, status, kind="good"):

    status_class = {
        "good": "good",
        "warn": "warn",
        "bad": "bad",
    }.get(kind, "good")

    st.markdown(
        f"""
        <div class="security-card {status_class}">
            <div class="security-icon">{icon}</div>
            <div class="security-name">{title}</div>
            <div class="security-status">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ============================================================
# SECURITY OVERVIEW
# ============================================================

def show_security_overview():

    integrity = model_integrity_ok()

    st.markdown(
        """
        <div class="section-kicker">
            Security layers
        </div>

        <div class="section-title">
            Active protection
        </div>

        <div class="section-subtitle">
            Security controls applied around the machine-learning
            prediction workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # MODEL INTEGRITY
    # --------------------------------------------------------

    if integrity:
        security_card(
            "🔐",
            "Model Integrity",
            "VERIFIED",
            "good",
        )
    else:
        security_card(
            "🔐",
            "Model Integrity",
            "BLOCKED",
            "bad",
        )

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    security_card(
        "🛡️",
        "Input Validation",
        "ACTIVE",
        "good",
    )

    # --------------------------------------------------------
    # ADVERSARIAL DEFENSE
    # --------------------------------------------------------

    security_card(
        "🧪",
        "Adversarial Stability",
        "ACTIVE",
        "good",
    )

    # --------------------------------------------------------
    # PRIVACY
    # --------------------------------------------------------

    security_card(
        "🔒",
        "Privacy Protection",
        "ACTIVE",
        "good",
    )

    # --------------------------------------------------------
    # ACTIVITY LOGGING
    # --------------------------------------------------------

    security_card(
        "📋",
        "Activity Logging",
        "ACTIVE",
        "good",
    )
# ============================================================
# SECURITY SCORE
# ============================================================

def show_security_score():
    integrity = model_integrity_ok()
    score = 100 if integrity else 0

    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-number">{score}</div>
            <div class="score-caption">Core model integrity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PATIENT INPUT FORM
# ============================================================

def show_prediction_form():
    st.markdown(
        """
        <div class="section-kicker">Clinical assessment</div>
        <div class="section-title">Patient information</div>
        <div class="section-subtitle">
            Enter the required clinical measurements. Patient medical
            information is used for prediction and is not stored in
            the activity log.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="form-shell">', unsafe_allow_html=True)

    st.markdown(
        '<div class="form-group-title">01  ·  Basic profile</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Age",
            min_value=-120,
            max_value=120,
            value=50,
            step=1,
        )

    with c2:
        sex = st.selectbox(
            "Sex",
            ["Female", "Male"],
        )

    with c3:
        cp = st.selectbox(
            "Chest Pain Type",
            [
                "asymptomatic",
                "atypical angina",
                "non-anginal",
                "typical angina",
            ],
        )

    st.markdown(
        '<div class="form-group-title">02  ·  Clinical measurements</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50.0,
            max_value=250.0,
            value=120.0,
            step=1.0,
        )

    with c2:
        chol = st.number_input(
            "Cholesterol",
            min_value=50.0,
            max_value=700.0,
            value=200.0,
            step=1.0,
        )

    with c3:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [False, True],
            format_func=lambda x: "Yes" if x else "No",
        )

    st.markdown(
        '<div class="form-group-title">03  ·  ECG & exercise response</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        restecg = st.selectbox(
            "Resting ECG",
            [
                "normal",
                "st-t abnormality",
                "lv hypertrophy",
            ],
        )

    with c2:
        thalch = st.number_input(
            "Maximum Heart Rate",
            min_value=50.0,
            max_value=250.0,
            value=150.0,
            step=1.0,
        )

    with c3:
        exang = st.selectbox(
            "Exercise Induced Angina",
            [False, True],
            format_func=lambda x: "Yes" if x else "No",
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        oldpeak = st.number_input(
            "ST Depression",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

    with c2:
        slope = st.selectbox(
            "ST Segment Slope",
            ["downsloping", "flat", "upsloping"],
        )

    with c3:
        ca = st.number_input(
            "Number of Major Vessels",
            min_value=0.0,
            max_value=4.0,
            value=0.0,
            step=1.0,
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        thal = st.selectbox(
            "Thalassemia",
            [
                "normal",
                "fixed defect",
                "reversable defect",
            ],
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "Generate Secure Prediction  →",
        type="primary",
        use_container_width=True,
    ):
        # Clear previous prediction before validating new input
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.anomaly_result = None
        st.session_state.stability_result = None

        run_prediction(
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalch,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION PIPELINE
# ============================================================

def run_prediction(
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalch,
    exang,
    oldpeak,
    slope,
    ca,
    thal,
):
    # --------------------------------------------------------
    # 1. Input validation
    # --------------------------------------------------------

    valid, error_message = validate_patient_data(
        age,
        trestbps,
        chol,
        thalch,
        oldpeak,
        ca,
    )

    if not valid:
        st.error(error_message)
        return

    patient_data = {
        "age": age,
        "sex": sex,
        "dataset": "Cleveland",
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    # --------------------------------------------------------
    # 2. Model integrity check
    # --------------------------------------------------------

    if not model_integrity_ok():
        st.error("MODEL SECURITY ERROR")
        st.warning(
            "The trusted ML model failed integrity verification. "
            "Prediction has been blocked."
        )
        return

    # --------------------------------------------------------
    # 3. Extraction protection
    # --------------------------------------------------------

    try:
        extraction_check = check_extraction_risk(
            st.session_state.username
        )
    except Exception:
        extraction_check = {
            "allowed": True,
            "message": "",
            "query_count": 0,
        }

    if not extraction_check["allowed"]:
        st.error("Prediction request blocked.")
        st.warning(extraction_check["message"])
        st.info(
            f"Recent prediction queries: "
            f"{extraction_check['query_count']}"
        )
        return

    # --------------------------------------------------------
    # 4. Input anomaly detection
    # --------------------------------------------------------

    try:
        feature_statistics = load_feature_statistics()

        anomaly_result = detect_input_anomaly(
            patient_data,
            feature_statistics,
        )
    except Exception:
        anomaly_result = {
            "suspicious": False,
            "features": [],
            "reasons": [],
        }

    st.session_state.anomaly_result = anomaly_result

    # --------------------------------------------------------
    # 5. Load trusted model
    # --------------------------------------------------------

    try:
        model = load_model()
    except RuntimeError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error("Unable to load the trusted model.")
        st.exception(error)
        return

    # --------------------------------------------------------
    # 6. Prediction stability
    # --------------------------------------------------------

    try:
        stability_result = check_prediction_stability(
            model,
            patient_data,
        )
    except Exception:
        stability_result = {
            "stable": True,
            "original_prediction": None,
            "changed_predictions": [],
        }

    st.session_state.stability_result = stability_result

    # --------------------------------------------------------
    # 7. Final prediction
    # --------------------------------------------------------

    try:
        prediction, confidence = predict_heart_disease(
            patient_data
        )
    except RuntimeError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)
        return

    # --------------------------------------------------------
    # 8. Save only prediction metadata
    # --------------------------------------------------------

    st.session_state.prediction_result = prediction
    st.session_state.confidence = confidence

    log_prediction(
        st.session_state.username,
        prediction,
        confidence,
    )

    st.session_state.page = "Prediction"
    st.toast("Prediction generated securely.")
    st.rerun()


# ============================================================
# PREDICTION RESULT
# ============================================================

def show_prediction_result():
    prediction = st.session_state.prediction_result
    confidence = st.session_state.confidence

    if prediction is None:
        return

    anomaly = st.session_state.anomaly_result or {
        "suspicious": False,
        "reasons": [],
    }

    stability = st.session_state.stability_result or {
        "stable": True,
    }

    st.markdown(
        """
        <div style="height:18px;"></div>
        <div class="section-kicker">Assessment complete</div>
        <div class="section-title">Prediction result</div>
        """,
        unsafe_allow_html=True,
    )

    if prediction == 1:
        result_class = "risk"
        result_title = "Heart Disease Risk Detected"
        result_icon = "⚠"
    else:
        result_class = "safe"
        result_title = "No Heart Disease Risk Detected"
        result_icon = "✓"

    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="result-label">AI assessment</div>
            <div class="result-title">{result_icon} {result_title}</div>
            <div class="confidence-value">{confidence:.2f}%</div>
            <div class="confidence-label">Model confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        anomaly_ok = not anomaly.get("suspicious", False)
        status = "PASSED" if anomaly_ok else "REVIEW"
        kind = "good" if anomaly_ok else "warn"

        security_card(
            "🛡️",
            "Input Anomaly",
            status,
            kind,
        )

    with c2:
        stable = stability.get("stable", True)
        status = "PASSED" if stable else "REVIEW"
        kind = "good" if stable else "warn"

        security_card(
            "🧪",
            "Prediction Stability",
            status,
            kind,
        )

    if anomaly.get("suspicious", False):
        st.warning(
            "Suspicious input characteristics were detected. "
            "Review the flagged characteristics before treating "
            "the prediction as reliable."
        )

        reasons = anomaly.get("reasons", [])
        for reason in reasons:
            st.write(f"• {reason}")

    if not stability.get("stable", True):
        st.warning(
            "This input is sensitive to small changes in one or "
            "more features. Additional clinical verification is "
            "recommended."
        )

    st.info(
        "Privacy notice: patient medical information is used only "
        "for prediction. The activity log stores username, "
        "prediction, confidence, and timestamp — not the patient's "
        "medical inputs."
    )

    st.caption(
        "Educational demonstration only. This prediction does not "
        "replace professional medical advice."
    )


# ============================================================
# SECURITY CENTER
# ============================================================
def show_security_overview():
    integrity = model_integrity_ok()

    st.markdown(
        """
        <div class="section-kicker">
            Security layers
        </div>

        <div class="section-title">
            Active protection
        </div>

        <div class="section-subtitle">
            Security controls applied around the machine-learning
            prediction workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if integrity:
        security_card(
            "🔐",
            "Model Integrity",
            "VERIFIED",
            "good"
        )
    else:
        security_card(
            "🔐",
            "Model Integrity",
            "BLOCKED",
            "bad"
        )

    security_card(
        "🛡️",
        "Input Validation",
        "ACTIVE",
        "good"
    )

    security_card(
        "🧪",
        "Adversarial Stability",
        "ACTIVE",
        "good"
    )

    security_card(
        "🔒",
        "Privacy Protection",
        "ACTIVE",
        "good"
    )

    security_card(
        "📋",
        "Activity Logging",
        "ACTIVE",
        "good"
    )
def show_security_center():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">SECURITY CENTER</div>
            <div class="hero-title">
                Protection around<br>
                <span>every prediction.</span>
            </div>
            <div class="hero-text">
                Secure Heart combines model integrity verification,
                input validation, adversarial defense, privacy-aware
                outputs, extraction protection, and activity logging
                into one ML security workflow.
            </div>
            <div class="hero-badge">● DEFENSE LAYERS ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.8, 1])

    with c1:
        show_security_overview()

    with c2:
        show_security_score()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-kicker">How it works</div>
        <div class="section-title">Prediction security flow</div>
        <div class="section-subtitle">
            The trusted model is protected before, during, and after
            prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    flow = [
        ("01", "Authenticate", "User identity is verified before access."),
        ("02", "Validate input", "Unexpected or invalid values are rejected or reviewed."),
        ("03", "Verify model", "The trusted model fingerprint is checked."),
        ("04", "Defend inference", "Extraction and prediction-stability checks run."),
        ("05", "Predict & log", "Only safe prediction metadata is recorded."),
    ]

    for number, title, text in flow:
        c1, c2 = st.columns([1, 6])

        with c1:
            st.markdown(
                f"""
                <div style="
                    color:#ff707a;
                    font-weight:850;
                    font-size:13px;
                    padding-top:3px;">
                    {number}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div style="
                    padding:0 0 18px;
                    border-bottom:1px solid rgba(255,255,255,.06);
                    margin-bottom:18px;">
                    <div style="font-weight:750;">{title}</div>
                    <div style="
                        color:#9299a8;
                        font-size:12px;
                        margin-top:5px;">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# ADMIN ANALYTICS
# ============================================================

def show_admin_analytics():
    if st.session_state.role != "admin":
        st.error("Admin access required.")
        return

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">ADMIN CONTROL CENTER</div>
            <div class="hero-title">
                Monitor the <span>secure system.</span>
            </div>
            <div class="hero-text">
                Review prediction activity, authentication history,
                and the current security state without exposing
                patient medical inputs.
            </div>
            <div class="hero-badge">● ADMIN ACCESS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prediction_logs = get_prediction_logs()
    login_logs = get_login_logs()

    total_predictions = len(prediction_logs)

    heart_count = sum(
        1 for log in prediction_logs if log[1] == 1
    )

    no_heart_count = sum(
        1 for log in prediction_logs if log[1] == 0
    )

    failed_logins = sum(
        1 for log in login_logs if log[1] == "FAILED"
    )

    cols = st.columns(4)

    metrics = [
        ("TOTAL PREDICTIONS", total_predictions),
        ("HEART DISEASE", heart_count),
        ("NO HEART DISEASE", no_heart_count),
        ("FAILED LOGINS", failed_logins),
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(
        ["Prediction Activity", "Login History"]
    )

    with tab1:
        if prediction_logs:
            data = []

            for log in prediction_logs:
                data.append(
                    {
                        "Username": log[0],
                        "Prediction": (
                            "Heart Disease"
                            if log[1] == 1
                            else "No Heart Disease"
                        ),
                        "Confidence": f"{log[2]:.2f}%",
                        "Timestamp": log[3],
                    }
                )

            history_df = pd.DataFrame(data)

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No prediction activity recorded yet.")

    with tab2:
        if login_logs:
            data = []

            for log in login_logs:
                data.append(
                    {
                        "Username": log[0],
                        "Status": (
                            "✓ Successful"
                            if log[1] == "SUCCESS"
                            else "✕ Failed"
                        ),
                        "Timestamp": log[2],
                    }
                )

            login_df = pd.DataFrame(data)

            st.dataframe(
                login_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No login activity recorded yet.")


# ============================================================
# PREDICTION PAGE
# ============================================================

def show_prediction_page():
    integrity = model_integrity_ok()

    if not integrity:
        st.error("MODEL SECURITY ERROR")
        st.warning(
            "The trusted ML model failed integrity verification. "
            "Prediction access has been blocked."
        )
        return

    username = st.session_state.username
    role = str(st.session_state.role).upper()

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">SECURE HEART · {role}</div>
            <div class="hero-title">
                Heart risk prediction,<br>
                <span>secured by design.</span>
            </div>
            <div class="hero-text">
                Welcome back, {username}. Enter patient information
                to generate a protected machine-learning prediction
                with security checks around the inference process.
            </div>
            <div class="hero-badge">● MODEL INTEGRITY VERIFIED</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_prediction_form()

    # ECG IMAGE ANALYSIS
    show_ecg_analysis()

    show_prediction_result()

# ============================================================
# MATRIX MANAGEMENT
# ============================================================

def show_matrix_management():
    if st.session_state.get("role") != "admin":
        st.error("Access denied. Admin privileges required.")
        return

    st.markdown("## 🔐 Matrix Management")

    st.info(
        "This page is available only to the administrator. "
        "Use the permanent matrix to provision physical security cards."
    )

    connection = sqlite3.connect("database/users.db")
    cursor = connection.cursor()

    cursor.execute(
        "SELECT username, role FROM users ORDER BY id"
    )

    accounts = cursor.fetchall()
    connection.close()

    if not accounts:
        st.warning("No user accounts found.")
        return

    usernames = [account[0] for account in accounts]

    selected_user = st.selectbox(
        "Select Account",
        usernames,
        key="matrix_management_user",
    )

    matrix = generate_matrix(selected_user)

    st.markdown("---")

    st.markdown(
        f"### 🪪 Permanent Security Matrix — `{selected_user}`"
    )

    st.caption(
        "This is the permanent matrix assigned to this account. "
        "Do not regenerate it when issuing a physical card."
    )

    st.markdown("#### Security Matrix")

    for row in matrix:
        columns = st.columns(4)

        for index, value in enumerate(row):
            columns[index].markdown(
                f"""
                <div style="
                    border: 2px solid #444;
                    border-radius: 10px;
                    padding: 18px;
                    text-align: center;
                    font-size: 26px;
                    font-weight: bold;
                    background-color: #f5f5f5;
                    color: #111;
                    margin-bottom: 10px;
                ">
                    {value}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.success(
        f"Permanent matrix loaded successfully for {selected_user}."
    )

    st.warning(
        "Keep this matrix confidential. It should only be given "
        "to the corresponding account holder as a physical security card."
    )


# ============================================================
# MAIN APP
# ============================================================

def main_application():
    if not check_session_timeout():
        return

    show_sidebar()

    page = st.session_state.page

    if page == "Prediction":
        show_prediction_page()

    elif page == "Security Center":
        show_security_center()

    elif page == "Admin Dashboard":
        show_admin_dashboard(
            get_prediction_logs,
            get_login_logs,
            model_integrity_ok,
        )

    elif page == "Matrix Management":
        show_matrix_management()


# ============================================================
# APPLICATION CONTROL
# ============================================================

if st.session_state.matrix_pending:
    show_matrix_verification()

elif st.session_state.face_pending:
    show_face_verification()

elif st.session_state.logged_in:
    main_application()

else:
    login_page()
    