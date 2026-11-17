import streamlit as st
from PIL import Image

from ecg.ecg_predict import (
    predict_ecg,
    get_ecg_interpretation
)


def show_ecg_analysis():
    """
    Display the ECG image upload and analysis section.

    This function is completely separate from the existing
    14-feature heart disease prediction system.
    """

    st.markdown(
        """
        <div class="ecg-section-header">
            <div class="ecg-section-icon">♥</div>
            <div>
                <h2>ECG Image Analysis</h2>
                <p>
                    Upload an ECG image to obtain an AI-based
                    ECG classification.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="ecg-info-box">
            <strong>How it works</strong><br>
            Upload a clear ECG image. The trained CNN model
            analyzes the image and classifies it into one of
            four ECG categories.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload ECG image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear ECG scan in JPG, JPEG, or PNG format."
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div class="ecg-upload-placeholder">
                <div class="upload-icon">+</div>
                <h3>Upload an ECG image</h3>
                <p>Supported formats: JPG, JPEG, PNG</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        return

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(uploaded_file).convert("RGB")

    except Exception as error:

        st.error(
            f"Unable to read the uploaded image: {error}"
        )

        return

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="ecg-subheading">ECG Preview</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Uploaded ECG",
        use_container_width=True
    )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze = st.button(
        "Analyze ECG Image",
        type="primary",
        use_container_width=True,
        key="analyze_ecg_button"
    )

    if not analyze:
        return

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner("Analyzing ECG image..."):

        try:

            result = predict_ecg(image)

            interpretation = get_ecg_interpretation(
                result["class_name"],
                result["confidence"]
            )

        except Exception as error:

            st.error(
                "ECG analysis failed."
            )

            st.caption(
                f"Technical details: {error}"
            )

            return

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    label = interpretation["label"]
    severity = interpretation["severity"]
    confidence = result["confidence"]

    if severity == "NORMAL":

        result_class = "ecg-result-normal"
        status_text = "NORMAL"

    elif severity == "ATTENTION":

        result_class = "ecg-result-attention"
        status_text = "ATTENTION"

    else:

        result_class = "ecg-result-review"
        status_text = "REVIEW"

    st.markdown(
        f"""
        <div class="{result_class}">
            <div class="ecg-result-top">
                <span class="ecg-result-label">
                    ECG Classification
                </span>

                <span class="ecg-status">
                    {status_text}
                </span>
            </div>

            <div class="ecg-result-name">
                {label}
            </div>

            <div class="ecg-confidence">
                <span>Model confidence</span>
                <strong>{confidence * 100:.1f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    st.markdown(
        '<div class="ecg-subheading">Class Probabilities</div>',
        unsafe_allow_html=True
    )

    class_names = [
        "Myocardial Infarction",
        "History of MI",
        "Abnormal Heartbeat",
        "Normal ECG"
    ]

    probabilities = result["probabilities"]

    for name, probability in zip(
        class_names,
        probabilities
    ):

        st.progress(
            float(probability),
            text=f"{name} — {probability * 100:.1f}%"
        )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="ecg-interpretation">
            <strong>{interpretation["message"]}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SAFETY NOTICE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="ecg-safety">
            <strong>Important:</strong>
            This is an AI-based image classification result,
            not a medical diagnosis. ECG findings should be
            reviewed by a qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True
    )