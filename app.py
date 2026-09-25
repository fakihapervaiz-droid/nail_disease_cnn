import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import os

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
CLASS_NAMES_PATH = "class_names.pkl"

IMG_SIZE = 224

# IMPORTANT:
# Replace this with the actual test accuracy from your model
MODEL_ACCURACY = 0.00


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
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
    padding: 20px;
    border-radius: 12px;
    background-color: #eef6ff;
    border: 1px solid #b9d9ff;
    text-align: center;
    margin-top: 20px;
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
    padding: 15px;
    border-radius: 10px;
    background-color: #f5f5f5;
    text-align: center;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# LOAD CLASS NAMES
# =========================================================
@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "rb") as f:
        return pickle.load(f)


# =========================================================
# MAIN UI
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
            <strong>Model Test Accuracy</strong><br>
            <span style="font-size:28px;">
                {MODEL_ACCURACY * 100:.2f}%
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info(
        "Set MODEL_ACCURACY in app.py to your actual test accuracy "
        "to display it here."
    )


# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a nail image",
    type=["jpg", "jpeg", "png", "webp"]
)


# =========================================================
# PREDICTION
# =========================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Your uploaded nail image",
        width=400
    )

    # ---------------------------------------------
    # PREPROCESS IMAGE
    # ---------------------------------------------
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image_resized)

    # Same preprocessing used by Simple CNN
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # ---------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------
    try:
        model = load_model()
        class_names = load_class_names()

        # ---------------------------------------------
        # PREDICTION
        # ---------------------------------------------
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index] * 100

        # ---------------------------------------------
        # RESULT
        # ---------------------------------------------
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

        # ---------------------------------------------
        # ALL CLASS PROBABILITIES
        # ---------------------------------------------
        st.subheader("Prediction Probabilities")

        for i, class_name in enumerate(class_names):

            probability = float(predictions[i])

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

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
    "This application is an AI/ML demonstration and is not a "
    "medical diagnosis. Predictions should not replace evaluation "
    "by a qualified healthcare professional."
)