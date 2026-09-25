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

# Replace this with your actual Simple CNN TEST accuracy.
# Example: 0.8245 means 82.45%
MODEL_ACCURACY = 0.8245


# =========================================================
# CLASS NAMES
# IMPORTANT:
# The order must match the class order used during training.
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
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #f8fafc,
            #eef2ff
        );
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }

    .hero h1 {
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }

    .hero p {
        color: #475569;
        font-size: 1.05rem;
    }

    .info-box {
        padding: 1.2rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0 1rem 0;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

        <h1>Nail Disease AI Classifier</h1>

        <p>
            An image classification application powered by a
            Simple Convolutional Neural Network (CNN).
            Upload a nail image to receive an AI-based prediction.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# =========================================================
# LOAD MODEL WITH ERROR HANDLING
# =========================================================

try:

    model = load_model()

except Exception as e:

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
        "This application uses a Simple CNN trained "
        "for nail disease image classification."
    )

    st.divider()

    st.metric(
        "Test Accuracy",
        f"{MODEL_ACCURACY * 100:.2f}%"
    )

    st.metric(
        "Image Size",
        f"{IMG_SIZE} × {IMG_SIZE}"
    )

    st.metric(
        "Number of Classes",
        len(CLASS_NAMES)
    )

    st.divider()

    st.subheader("Supported Classes")

    for class_name in CLASS_NAMES:

        st.write(
            f"• {class_name}"
        )


# =========================================================
# MAIN LAYOUT
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


    # -----------------------------------------------------
    # SHOW UPLOADED IMAGE
    # -----------------------------------------------------

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
    # PREPROCESS IMAGE
    # -----------------------------------------------------

    image_resized = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        image_resized
    ).astype("float32")

    # Same preprocessing used during training
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    # -----------------------------------------------------
    # FIND PREDICTED CLASS
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
    # SHOW RESULT
    # =====================================================

    with result_column:

        st.subheader(
            "Prediction Result"
        )

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
        # CONFIDENCE INTERPRETATION
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
                "Consider using a clearer image."
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

        probability_data[
            "Probability (%)"
        ] = (
            probability_data["Probability"] * 100
        ).round(2)


        # Sort from highest to lowest
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
# IMAGE PROCESSING INFORMATION
# =========================================================

if uploaded_file is not None:

    st.divider()

    with st.expander(
        "Image Processing Details"
    ):

        st.write(
            "The uploaded image is processed using "
            "the same basic preprocessing pipeline used "
            "during model training."
        )

        st.write(
            f"• Original image: "
            f"{image.width} × {image.height}"
        )

        st.write(
            f"• Resized image: "
            f"{IMG_SIZE} × {IMG_SIZE}"
        )

        st.write(
            "• Color format: RGB"
        )

        st.write(
            "• Pixel normalization: 0–1 "
            "using division by 255"
        )

        st.write(
            "• Model input shape: "
            f"(1, {IMG_SIZE}, {IMG_SIZE}, 3)"
        )


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.subheader(
    "How It Works"
)

step1, step2, step3 = st.columns(3)


with step1:

    st.markdown(
        """
        **1. Upload**

        Upload a clear nail image using
        the image uploader.
        """
    )


with step2:

    st.markdown(
        """
        **2. Preprocess**

        The image is resized to 224×224
        pixels and normalized.
        """
    )


with step3:

    st.markdown(
        """
        **3. Predict**

        The trained Simple CNN analyzes
        the image and produces probabilities
        for six classes.
        """
    )


# =========================================================
# MODEL DETAILS
# =========================================================

st.divider()

st.subheader(
    "Model Architecture"
)

model_info_column, accuracy_column = st.columns(
    [2, 1]
)


with model_info_column:

    st.write(
        """
        The classifier is based on a custom
        Convolutional Neural Network containing:

        - Convolutional layers
        - Batch Normalization
        - Max Pooling
        - Global Average Pooling
        - Fully Connected layer
        - Dropout
        - Softmax output layer
        """
    )


with accuracy_column:

    st.metric(
        "Simple CNN Test Accuracy",
        f"{MODEL_ACCURACY * 100:.2f}%"
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.warning(
    """
    **Educational Use Only**

    This application is an AI-based image classification
    project and should not be used as a medical diagnosis
    or a replacement for professional medical advice.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        Nail Disease AI Classifier<br>
        Built with TensorFlow and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)