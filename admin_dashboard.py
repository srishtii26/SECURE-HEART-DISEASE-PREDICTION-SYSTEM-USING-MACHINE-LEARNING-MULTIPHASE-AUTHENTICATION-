import streamlit as st
import pandas as pd


def show_admin_dashboard(get_prediction_logs, get_login_logs, model_integrity_ok):
    """
    Display the Admin Dashboard.

    Admin-only protection is enforced here as well as in app.py.
    """

    # ---------------------------------------------------------
    # ADMIN ACCESS PROTECTION
    # ---------------------------------------------------------
    if st.session_state.get("role") != "admin":
        st.error("🚫 Admin access required.")
        return

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">ADMIN CONTROL CENTER</div>
            <div class="hero-title">
                Monitor the <span>secure system.</span>
            </div>
            <div class="hero-text">
                Review system activity, authentication history,
                and machine-learning security status.
            </div>
            <div class="hero-badge">● ADMIN ACCESS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # GET LOGS
    # ---------------------------------------------------------
    prediction_logs = get_prediction_logs()
    login_logs = get_login_logs()

    total_predictions = len(prediction_logs)

    heart_count = sum(
        1 for log in prediction_logs
        if log[1] == 1
    )

    no_heart_count = sum(
        1 for log in prediction_logs
        if log[1] == 0
    )

    failed_logins = sum(
        1 for log in login_logs
        if log[1] == "FAILED"
    )

    # ---------------------------------------------------------
    # MODEL INTEGRITY
    # ---------------------------------------------------------
    integrity = model_integrity_ok()

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # SECURITY STATUS
    # ---------------------------------------------------------
    st.markdown("### 🛡️ System Security")

    security_col1, security_col2 = st.columns(2)

    with security_col1:
        if integrity:
            st.success("✅ ML Model Integrity: VERIFIED")
        else:
            st.error("🚨 ML Model Integrity: FAILED")

    with security_col2:
        st.success("✅ Role-Based Access Control: ACTIVE")

    # ---------------------------------------------------------
    # ACTIVITY TABS
    # ---------------------------------------------------------
    tab1, tab2 = st.tabs(
        ["📊 Prediction Activity", "🔐 Login History"]
    )

    # ---------------------------------------------------------
    # PREDICTION ACTIVITY
    # ---------------------------------------------------------
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
            st.info(
                "No prediction activity recorded yet."
            )

    # ---------------------------------------------------------
    # LOGIN HISTORY
    # ---------------------------------------------------------
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
            st.info(
                "No login activity recorded yet."
            )