import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import time


def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()


# Load the trained models
vgg16_model = tf.keras.models.load_model("vgg16_model.keras")
densenet_model = tf.keras.models.load_model("densenet_model.keras")

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

    # VGG16 Predictions
    vgg16_predictions = vgg16_model.predict(processed_img)
    vgg16_class_index = np.argmax(vgg16_predictions, axis=1)
    vgg16_confidence_scores = vgg16_predictions[0]

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
    vgg_start = time.time()
    vgg16_predicted_label = labels[vgg16_class_index[0]]
    vgg16_confidence = vgg16_confidence_scores[vgg16_class_index[0]]
    vgg16_confidence_percentage = vgg16_confidence * 100  # Convert to percentage

    # Grad-CAM Visualization for VGG16
    # Adjust to the last conv layer of your VGG16 model
    last_conv_layer_name_vgg16 = 'block5_conv3'
    vgg16_heatmap = make_gradcam_heatmap(
        processed_img, vgg16_model, last_conv_layer_name_vgg16)
    img_array = np.array(img)
    vgg16_gradcam_img = overlay_gradcam(img_array, vgg16_heatmap)
    vgg_stop = time.time()
    vgg_total = round(vgg_stop - vgg_start, 3)
    
    # Create two columns for the output
    col1, col2 = st.columns(2)

    # VGG16 Output
    with col1:
        st.header("VGG16 Result")
        st.image(vgg16_gradcam_img, caption="VGG16 Grad-CAM",
                 use_column_width=True)
        st.write(f"Predicted Label: {vgg16_predicted_label}")
        st.write(f"Confidence Score: {vgg16_confidence:.4f} or {
                 vgg16_confidence_percentage:.2f}%")
        st.header(f"Total time take to predict: {vgg_total} Seconds")
        st.write(" ")
        st.write(" ")
        st.title("Confidence Scores for all classes (VGG16):")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        for i, label in labels.items():
            score = vgg16_confidence_scores[i]
            percentage = score * 100
            st.write(f"{label}: {score:.4f} or {percentage:.2f}%")


    # DenseNet Predictions
    dense_start = time.time()
    densenet_predictions = densenet_model.predict(processed_img)
    densenet_class_index = np.argmax(densenet_predictions, axis=1)
    densenet_confidence_scores = densenet_predictions[0]

    densenet_predicted_label = labels[densenet_class_index[0]]
    densenet_confidence = densenet_confidence_scores[densenet_class_index[0]]
    densenet_confidence_percentage = densenet_confidence * 100  # Convert to percentage

    # Grad-CAM Visualization for DenseNet
    last_conv_layer_name_densenet = 'conv5_block16_concat'  # Adjust for DenseNet121
    densenet_heatmap = make_gradcam_heatmap(
        processed_img, densenet_model, last_conv_layer_name_densenet)
    densenet_gradcam_img = overlay_gradcam(img_array, densenet_heatmap)
    dense_stop = time.time()
    dense_total = round(dense_stop - dense_start, 3)


    # DenseNet121 Output
    with col2:
        st.header("DenseNet121 Result")
        st.image(densenet_gradcam_img, caption="DenseNet121 Grad-CAM",
                 use_column_width=True)
        st.write(f"Predicted Label: {densenet_predicted_label}")
        st.write(f"Confidence Score: {densenet_confidence:.4f} or {
                 densenet_confidence_percentage:.2f}%")
        st.header(f"Total time take to predict: {dense_total} Seconds")
        st.write(" ")
        st.title("Confidence Scores for all classes (DenseNet121):")
        for i, label in labels.items():
            score = densenet_confidence_scores[i]
            percentage = score * 100
            st.write(f"{label}: {score:.4f} or {percentage:.2f}%")
