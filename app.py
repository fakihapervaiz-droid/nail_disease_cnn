import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Nail Disease AI Classifier",
    page_icon="🩺",
    layout="centered"
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "nail_disease_simple_cnn.keras"

IMG_SIZE = 224

# ---------------------------------------------------------
# PUT YOUR ACTUAL TEST ACCURACY HERE
# Example:
# If your test accuracy is 82.45%, write 0.8245
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

    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 22px;
        border-radius: 12px;
        background-color: #eef6ff;
        border: 1px solid #b9d9ff;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .prediction-title {
        font-size: 15px;
        color: #555;
    }

    .prediction-name {
        font-size: 30px;
        font-weight: 700;
        margin: 8px 0;
    }

    .confidence {
        font-size: 18px;
        font-weight: 600;
    }

    .accuracy-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #f5f5f5;
        text-align: center;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f8f8;
        margin-top: 20px;
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
    '<div class="title">Nail Disease AI Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a nail image and let the trained CNN model analyze it.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL ACCURACY
# =========================================================

if MODEL_ACCURACY > 0:

    st.markdown(
        f"""
        <div class="accuracy-box">

            <strong>Model Test Accuracy</strong>

            <br><br>

            <span style="font-size:30px;font-weight:700;">
                {MODEL_ACCURACY * 100:.2f}%
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.info(
        "Enter your actual test accuracy in MODEL_ACCURACY "
        "inside app.py to display it here."
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("Upload Nail Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # OPEN USER IMAGE
        # -------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # -------------------------------------------------
        # DISPLAY ORIGINAL IMAGE
        # -------------------------------------------------

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Your uploaded nail image",
            width=400
        )


        # -------------------------------------------------
        # PREPROCESS IMAGE
        # -------------------------------------------------

        image_resized = image.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        image_array = np.array(
            image_resized
        )

        image_array = image_array.astype(
            "float32"
        ) / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # -------------------------------------------------
        # LOAD MODEL
        # -------------------------------------------------

        model = load_model()


        # -------------------------------------------------
        # MAKE PREDICTION
        # -------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]


        # -------------------------------------------------
        # FIND PREDICTED CLASS
        # -------------------------------------------------

        predicted_index = np.argmax(
            predictions
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = (
            predictions[predicted_index] * 100
        )


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="prediction-box">

                <div class="prediction-title">
                    Predicted Condition
                </div>

                <div class="prediction-name">
                    {predicted_class}
                </div>

                <div class="confidence">
                    Confidence: {confidence:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # ALL PREDICTION PROBABILITIES
        # -------------------------------------------------

        st.subheader(
            "Prediction Probabilities"
        )

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = float(
                predictions[i]
            )

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )


        # -------------------------------------------------
        # IMAGE PROCESSING INFORMATION
        # -------------------------------------------------

        st.markdown(
            """
            <div class="info-box">

            <strong>Image Processing</strong>

            <br><br>

            Your uploaded image is automatically:

            <br>
            • Converted to RGB
            <br>
            • Resized to 224 × 224 pixels
            <br>
            • Normalized to values between 0 and 1
            <br>
            • Sent to the trained Simple CNN model

            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            "The model could not be loaded or prediction failed."
        )

        st.exception(e)


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "This application is an AI/ML demonstration and is not "
    "a medical diagnosis. Predictions should not replace "
    "evaluation by a qualified healthcare professional."
)