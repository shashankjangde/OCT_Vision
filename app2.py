import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

# loading the model
model = tf.keras.models.load_model("train.keras")

# Function to preprocess the input image


def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Function to generate Grad-CAM heatmap


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute the gradients of the top predicted class for the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Compute the guided gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps with the computed gradients
    conv_outputs = conv_outputs[0]
    conv_outputs = conv_outputs @ pooled_grads[..., tf.newaxis]
    conv_outputs = tf.squeeze(conv_outputs)

    # Normalize the heatmap
    heatmap = tf.maximum(conv_outputs, 0) / tf.math.reduce_max(conv_outputs)
    return heatmap.numpy()

# Function to overlay the Grad-CAM heatmap on the image


def overlay_gradcam(img, heatmap, alpha=0.4):
    # Resize heatmap to the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert the heatmap to RGB format
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap on the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

    return superimposed_img


# Streamlit UI
st.title("OCT Scan Classification with Grad-CAM")

# File uploader for predictions
uploaded_file = st.file_uploader(
    "Choose an OCT scan...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, color_mode="rgb")  # Load image
    st.image(img, caption="Uploaded OCT scan", use_column_width=True)

    # Preprocess the image
    processed_img = preprocess_image(img)

    # Make prediction
    predictions = model.predict(processed_img)
    class_index = np.argmax(predictions, axis=1)
    confidence_scores = predictions[0]  # Get confidence scores for each class

    # Mapping of class indices to disease labels
    labels = {
        0: "Age related Macular Degeneration - AMD",
        1: "Diabetic Macular Edema - DME",
        2: "Epiretinal Membrane - ERM",
        3: "Macular Neovascular membranes - NO",
        4: "Retinal Artery Occlusion - RAO",
        5: "Retinal Vein Occlusion - RVO",
        6: "Vitreomacular Interface Disease - VID",
    }

    predicted_label = labels[class_index[0]]
    # Confidence score of the predicted class
    confidence = confidence_scores[class_index[0]]
    confidence_percentage = confidence * 100  # Convert to percentage

    # Grad-CAM Visualization
    # Adjust to the last conv layer of your model
    last_conv_layer_name = 'block5_conv3'

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(processed_img, model, last_conv_layer_name)

    # Convert original image to array
    img_array = np.array(img)

    # Overlay Grad-CAM on the original image
    gradcam_img = overlay_gradcam(img_array, heatmap)

    # Display the Grad-CAM image
    st.title("Grad-CAM Visualization")
    st.image(gradcam_img, caption="Grad-CAM", use_column_width=True)

    # Display the result
    st.write(f"Predicted Label: {predicted_label}")
    st.write(f"Confidence Score: {confidence:.4f} or {
             confidence_percentage:.2f}%")

    # Display confidence scores for all classes
    st.write("")
    st.title("Confidence Scores for all classes:")
    for i, label in labels.items():
        score = confidence_scores[i]
        percentage = score * 100
        st.write(f"{label}: {score:.4f} or {percentage:.2f}%")


# Color Representation:
# Red/Orange/Yellow: These are the most important regions for the prediction. The model is strongly focusing on these areas to make its decision.
# Green: Represents regions with moderate importance. These areas contribute somewhat to the prediction, but not as much as the red/orange/yellow areas.
# Blue: Represents low importance regions. These areas contribute very little or nothing to the model's prediction.
