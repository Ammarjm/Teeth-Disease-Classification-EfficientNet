import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore


model = tf.keras.models.load_model(
    "best_efficientnet_teeth.keras"
)


class_names = [
    "CaS",
    "CoS",
    "Gum",
    "MC",
    "OC",
    "OLP",
    "OT"
]


st.title("Teeth Disease Classification")


uploaded_file = st.file_uploader(
    "Upload dental image",
    type=["jpg","png","jpeg"]
)


if uploaded_file:

    img = image.load_img(
        uploaded_file,
        target_size=(224,224)
    )

    st.image(
        img,
        caption="Uploaded Image"
    )


    # Convert image to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # EfficientNet preprocessing
    img_array = preprocess_input(
        img_array
    )


    prediction = model.predict(
        img_array
    )


    probabilities = prediction[0]

    predicted_class = class_names[
        np.argmax(probabilities)
    ]

    confidence = np.max(probabilities)


    st.success(
        f"Prediction: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )


    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):
        st.write(
            f"{class_name}: {probabilities[i]:.2%}"
        )