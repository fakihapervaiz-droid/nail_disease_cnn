import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Nail Disease AI Classifier",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "nail_disease_simple_cnn.keras"

IMG_SIZE = 224


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
# HEADER
# =========================================================

st.title("Nail Disease AI Classifier")

st.write(
    "An image classification application powered by a "
    "Simple Convolutional Neural Network (CNN). "
    "Upload a nail image to receive an AI-based prediction."
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


try:

    model = load_model()

except Exception:

    st.error(
        "Unable to load the trained model."
    )

    st.info(
        "Make sure 'nail_disease_simple_cnn.keras' "
        "is in the same folder as app.py."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Model Information")

    st.write(
        "This application uses a Simple CNN "
        "for nail disease image classification."
    )

    st.divider()

    st.write("**Input Size**")

    st.write(
        f"{IMG_SIZE} × {IMG_SIZE} pixels"
    )

    st.write("**Number of Classes**")

    st.write(
        str(len(CLASS_NAMES))
    )

    st.divider()

    st.subheader("Supported Classes")

    for class_name in CLASS_NAMES:

        st.write(
            f"• {class_name}"
        )


# =========================================================
# MAIN COLUMNS
# =========================================================

upload_column, result_column = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

with upload_column:

    st.subheader("Upload Nail Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        help="Upload a clear image of a nail."
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.caption(
            f"Original image size: "
            f"{image.width} × {image.height}"
        )


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # RESIZE IMAGE
    # -----------------------------------------------------

    image_resized = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )


    # -----------------------------------------------------
    # CONVERT IMAGE TO NUMPY
    # -----------------------------------------------------

    image_array = np.array(
        image_resized
    ).astype("float32")


    # -----------------------------------------------------
    # NORMALIZATION
    # Same preprocessing used during training
    # -----------------------------------------------------

    image_array = image_array / 255.0


    # -----------------------------------------------------
    # ADD BATCH DIMENSION
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    # -----------------------------------------------------
    # GET PREDICTED CLASS
    # -----------------------------------------------------

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    confidence_percent = confidence * 100


    # =====================================================
    # RESULT
    # =====================================================

    with result_column:

        st.subheader("Prediction Result")

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence_percent:.2f}%"
        )

        st.progress(
            confidence
        )


        # -------------------------------------------------
        # CONFIDENCE MESSAGE
        # -------------------------------------------------

        if confidence >= 0.80:

            st.success(
                "The model has high confidence in this prediction."
            )

        elif confidence >= 0.50:

            st.warning(
                "The model has moderate confidence in this prediction."
            )

        else:

            st.info(
                "The model has relatively low confidence. "
                "A clearer image may produce a different result."
            )


        # =================================================
        # CLASS PROBABILITIES
        # =================================================

        st.subheader(
            "Class Probabilities"
        )

        probability_data = pd.DataFrame(
            {
                "Condition": CLASS_NAMES,
                "Probability": predictions
            }
        )

        probability_data["Probability (%)"] = (
            probability_data["Probability"] * 100
        ).round(2)


        probability_data = probability_data.sort_values(
            by="Probability",
            ascending=False
        ).reset_index(drop=True)


        # -------------------------------------------------
        # DISPLAY PROBABILITIES
        # -------------------------------------------------

        for _, row in probability_data.iterrows():

            condition = row["Condition"]

            probability = float(
                row["Probability"]
            )

            percentage = float(
                row["Probability (%)"]
            )

            st.write(
                f"**{condition}** — {percentage:.2f}%"
            )

            st.progress(
                probability
            )


# =========================================================
# IMAGE PROCESSING DETAILS
# =========================================================

if uploaded_file is not None:

    st.divider()

    with st.expander(
        "Image Processing Details"
    ):

        st.write(
            "The uploaded image is processed using "
            "the same preprocessing approach used "
            "during model training."
        )

        st.write(
            f"Original image: "
            f"{image.width} × {image.height}"
        )

        st.write(
            f"Resized image: "
            f"{IMG_SIZE} × {IMG_SIZE}"
        )

        st.write(
            "Color format: RGB"
        )

        st.write(
            "Pixel normalization: 0–1"
        )

        st.write(
            "Model input shape: "
            f"(1, {IMG_SIZE}, {IMG_SIZE}, 3)"
        )


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.subheader("How It Works")

step1, step2, step3 = st.columns(3)


with step1:

    st.markdown("**1. Upload**")

    st.write(
        "Upload a clear nail image."
    )


with step2:

    st.markdown("**2. Preprocess**")

    st.write(
        "The image is resized to 224 × 224 "
        "pixels and normalized."
    )


with step3:

    st.markdown("**3. Predict**")

    st.write(
        "The Simple CNN analyzes the image "
        "and generates probabilities for "
        "six classes."
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("Model Architecture")

st.write(
    """
    The classifier is based on a custom Simple CNN
    containing convolutional layers, batch normalization,
    max pooling, global average pooling, a dense layer,
    dropout, and a softmax output layer.
    """
)


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.warning(
    "Educational Use Only: This application is an "
    "AI-based image classification project and should "
    "not be used as a medical diagnosis or a replacement "
    "for professional medical advice."
)


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "Nail Disease AI Classifier • Built with TensorFlow and Streamlit"
)