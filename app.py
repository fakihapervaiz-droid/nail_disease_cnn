import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NailCare AI | Nail Disease Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_PATH = "nail_disease_simple_cnn.keras"

IMG_SIZE = 224

# ---------------------------------------------------------
# IMPORTANT:
# Replace 0.00 with your ACTUAL Simple CNN test accuracy.
#
# Example:
# If your test accuracy = 82.45%
# write:
# MODEL_ACCURACY = 0.8245
# ---------------------------------------------------------

MODEL_ACCURACY = 0.00


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "Acral_Lentiginous_Melanoma",
    "Healthy_Nail",
    "Onychogryphosis",
    "blue_finger",
    "clubbing",
    "pitting"
]


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------
       GENERAL
    ------------------------------------------------- */

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* -------------------------------------------------
       HEADER
    ------------------------------------------------- */

    .hero {
        text-align: center;
        padding: 20px 10px 35px 10px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #6b7280;
        max-width: 720px;
        margin: auto;
        line-height: 1.6;
    }


    /* -------------------------------------------------
       INFO CARDS
    ------------------------------------------------- */

    .info-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        min-height: 115px;
    }

    .info-label {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .info-value {
        font-size: 25px;
        font-weight: 700;
    }


    /* -------------------------------------------------
       RESULT CARD
    ------------------------------------------------- */

    .result-card {
        background: #f8fbff;
        border: 1px solid #cfe3ff;
        border-radius: 16px;
        padding: 25px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .result-label {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 7px;
    }

    .result-name {
        font-size: 29px;
        font-weight: 750;
        line-height: 1.25;
        margin-bottom: 8px;
    }

    .result-confidence {
        font-size: 17px;
        color: #475569;
    }


    /* -------------------------------------------------
       SECTION TITLES
    ------------------------------------------------- */

    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }


    /* -------------------------------------------------
       FOOTER
    ------------------------------------------------- */

    .footer-note {
        color: #6b7280;
        font-size: 13px;
        line-height: 1.6;
        text-align: center;
        margin-top: 30px;
    }


    /* -------------------------------------------------
       UPLOADER
    ------------------------------------------------- */

    [data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            NailCare AI
        </div>

        <div class="hero-subtitle">
            AI-powered nail image classification using a
            trained Convolutional Neural Network.
            Upload a nail image to receive a model prediction.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Model
            </div>

            <div class="info-value">
                Simple CNN
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Image Size
            </div>

            <div class="info-value">
                224 × 224
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    accuracy_text = (
        f"{MODEL_ACCURACY * 100:.2f}%"
        if MODEL_ACCURACY > 0
        else "Add Accuracy"
    )

    st.markdown(
        f"""
        <div class="info-card">

            <div class="info-label">
                Test Accuracy
            </div>

            <div class="info-value">
                {accuracy_text}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SPACING
# =========================================================

st.write("")


# =========================================================
# MODEL ACCURACY MESSAGE
# =========================================================

if MODEL_ACCURACY <= 0:

    st.info(
        "Set MODEL_ACCURACY in app.py to your actual "
        "Simple CNN test accuracy."
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">Upload a Nail Image</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a JPG, JPEG, PNG, or WEBP image of a nail."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    label_visibility="collapsed"
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # LOAD USER IMAGE
        # -------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # -------------------------------------------------
        # DISPLAY IMAGE
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">Image Analysis</div>',
            unsafe_allow_html=True
        )

        image_col, result_col = st.columns(
            [1, 1],
            gap="large"
        )


        # =================================================
        # IMAGE COLUMN
        # =================================================

        with image_col:

            st.image(
                image,
                caption="Uploaded nail image",
                use_container_width=True
            )


        # =================================================
        # PREPROCESS IMAGE
        # =================================================

        image_resized = image.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        image_array = np.array(
            image_resized
        ).astype(
            "float32"
        ) / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # =================================================
        # LOAD MODEL
        # =================================================

        model = load_model()


        # =================================================
        # MAKE PREDICTION
        # =================================================

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


        # =================================================
        # GET RESULT
        # =================================================

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index]
        )

        confidence_percent = (
            confidence * 100
        )


        # =================================================
        # RESULT COLUMN
        # =================================================

        with result_col:

            st.markdown(
                """
                <div class="section-title">
                    Prediction Result
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        Predicted Condition
                    </div>

                    <div class="result-name">
                        {predicted_class}
                    </div>

                    <div class="result-confidence">
                        Model confidence:
                        <strong>
                            {confidence_percent:.2f}%
                        </strong>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                confidence
            )


        # =================================================
        # PROBABILITY RESULTS
        # =================================================

        st.markdown(
            '<div class="section-title">Class Probabilities</div>',
            unsafe_allow_html=True
        )

        probability_data = []

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability_data.append(
                {
                    "Condition": class_name,
                    "Probability": predictions[i] * 100
                }
            )

        probability_df = pd.DataFrame(
            probability_data
        )

        probability_df = probability_df.sort_values(
            by="Probability",
            ascending=False
        )

        for _, row in probability_df.iterrows():

            class_name = row["Condition"]

            probability = float(
                row["Probability"]
            )

            st.write(
                f"**{class_name}** — "
                f"{probability:.2f}%"
            )

            st.progress(
                probability / 100
            )


        # =================================================
        # PROCESSING DETAILS
        # =================================================

        with st.expander(
            "View image processing details"
        ):

            st.write(
                "The uploaded image is processed using "
                "the same basic preprocessing pipeline used "
                "for the Simple CNN."
            )

            st.write(
                "Original image size:",
                image.size
            )

            st.write(
                "Model input size:",
                "224 × 224"
            )

            st.write(
                "Color format:",
                "RGB"
            )

            st.write(
                "Pixel normalization:",
                "0–255 → 0–1"
            )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error(
            "The model could not be loaded or the image "
            "could not be processed."
        )

        st.exception(e)


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)

step1, step2, step3 = st.columns(3)

with step1:

    st.markdown(
        """
        **1. Upload**

        Upload your own nail image using
        the image uploader.
        """
    )

with step2:

    st.markdown(
        """
        **2. Preprocess**

        The image is converted to RGB,
        resized to 224 × 224, and normalized.
        """
    )

with step3:

    st.markdown(
        """
        **3. Predict**

        The trained Simple CNN analyzes
        the image and returns probabilities
        for six classes.
        """
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer-note">

    <strong>Important:</strong>
    This application is an AI/ML project for
    educational and demonstration purposes.
    It is not a medical diagnostic tool.
    Model predictions should not be used as a
    substitute for professional medical evaluation.

    </div>
    """,
    unsafe_allow_html=True
)