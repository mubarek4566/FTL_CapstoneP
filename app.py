import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Define dataset path
DATASET_PATH = 'C:\\Users\\Specter\\Desktop\\Friontier ML training\\capstone_project\\code\\data\\PlantVillage'

@st.cache_data
def load_image_data(dataset_path):
    image_data = []
    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_name)
                    image_data.append({'label': label, 'path': img_path})
    return pd.DataFrame(image_data)

# Constants
IMG_SIZE = 224
MODEL_PATH = "plant_disease_detector3.keras"

# Class labels and tooltips
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

disease_info = {
    'Pepper__bell___Bacterial_spot': 'Bacterial infection with black/brown spots on pepper.',
    'Pepper__bell___healthy': 'Healthy pepper leaf.',
    'Potato___Early_blight': 'Fungal disease; dark concentric rings on leaves.',
    'Potato___Late_blight': 'Fungal; causes rapid foliage and tuber rot.',
    'Potato___healthy': 'Healthy potato leaf.',
    'Tomato_Bacterial_spot': 'Bacterial lesions on leaves and fruits.',
    'Tomato_Early_blight': 'Target-like brown fungal spots.',
    'Tomato_Late_blight': 'Aggressive rot on leaves and fruit.',
    'Tomato_Leaf_Mold': 'Yellow patches and mold on underside of leaves.',
    'Tomato_Septoria_leaf_spot': 'Small round spots with dark borders.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tiny pests; yellow speckling.',
    'Tomato__Target_Spot': 'Dark concentric fungal rings on fruit and leaves.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Virus; leaf curling and yellowing.',
    'Tomato__Tomato_mosaic_virus': 'Mottled coloring, leaf curling.',
    'Tomato_healthy': 'Healthy tomato leaf.'
}

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Page config
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="wide")
st.markdown("<h1 style='text-align: center;'>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 EDA"])

with tab2:
    # Load data
    df = load_image_data(DATASET_PATH)

    # Assuming df1 is already defined (e.g., loaded from your dataset)
    st.markdown("### 📊 Image Count per Class")

    # Create two columns
    col1, col2 = st.columns(2)

    # Function to create the bar plot for class distribution
    def get_class_distribution_plot(df):
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.countplot(
            data=df,
            x='label',
            order=df['label'].value_counts().index,
            ax=ax
        )
        ax.set_title("Class Distribution: Healthy vs Disease Types")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Images")
        plt.xticks(rotation=90)
        return fig
    

    def get_sampled_image_properties(df, sample_size=3000):
        sizes = []
        brightness = []

        # st.write(f"🔍 Analyzing image properties (sample of up to {sample_size} images)...")
        sampled_paths = df['path'].sample(min(sample_size, len(df))).tolist()

        for path in tqdm(sampled_paths):
            try:
                img = Image.open(path).convert('L')  # Convert to grayscale
                sizes.append(img.size)  # (width, height)
                brightness.append(np.mean(np.array(img)))
            except Exception as e:
                continue

        # Create DataFrame from results
        sizes_df = pd.DataFrame(sizes, columns=['width', 'height'])
        sizes_df['brightness'] = brightness
        return sizes_df

    # --- Function to generate brightness plot ---
    def get_brightness_distribution_plot(sizes_df):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            sizes_df['brightness'],
            bins=50,
            color='green',
            kde=True,
            ax=ax
        )
        ax.set_title("Brightness Distribution of Images")
        ax.set_xlabel("Average Pixel Brightness (0–255)")
        fig.tight_layout()
        return fig

    def display_brightness_analysis(df, sample_size=3000):
        sizes_df = get_sampled_image_properties(df, sample_size)
        if sizes_df.empty:
            st.warning("No valid image data was found.")
            return

        # st.markdown("## 🌞 Brightness Distribution (Sampled Images)")
        fig = get_brightness_distribution_plot(sizes_df)
        return fig
       

    
    
    # Display the plot in both columns
    with col1:
        st.pyplot(get_class_distribution_plot(df))

    with col2:
        st.pyplot(display_brightness_analysis(df))

    st.markdown("### 📁 Browse Sample Images by Class")

    # Create mapping from short_label to full class name
    short_label_map = {class_name.split("___")[-1].replace("_", " "): class_name for class_name in class_names}
    short_labels = list(short_label_map.keys())

    # Show dropdown and use first label as default
    selected_short_label = st.selectbox("🔍 Select a Class to View Samples", short_labels)

    # Get the corresponding full class name
    selected_class_name = short_label_map[selected_short_label]
    # st.write(selected_class_name)

    # Folder path
    folder_path = f"Sample_Image/{selected_class_name}/"

    # Check and load all .jpg images from the folder
    if os.path.exists(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

        # Display images in rows of 3
        for i in range(0, len(image_files), 3):
            image_cols = st.columns(3)
            for j, col in enumerate(image_cols):
                if i + j < len(image_files):
                    image_path = os.path.join(folder_path, image_files[i + j])
                    with col:
                        st.image(image_path, use_container_width=True)
    else:
        st.warning("Image folder not found", icon="⚠️")

    # Description below images
    st.markdown(disease_info.get(selected_class_name, "No description available."))
    # st.caption(disease_info.get(selected_class_name, "No description available."))



with tab1:
    #st.write("Upload a clear leaf image to detect the disease.")
    uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"], key="unique_uploader")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = np.max(prediction)

        st.markdown(f"### 🧪 Prediction: **{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
        st.info(disease_info.get(predicted_class, "No description available."))

        # Top-N dropdown
        top_n = st.selectbox("🔬 Select top-N class probabilities to view:", [3, 5, 7, 10, "All"])
        sorted_indices = np.argsort(prediction)[::-1]
        sorted_classes = [(class_names[i], prediction[i]) for i in sorted_indices]

        if top_n != "All":
            sorted_classes = sorted_classes[:int(top_n)]

        st.markdown("### 📊 Top Class Probabilities")
        for cls, prob in sorted_classes:
            st.write(f"{cls}: **{prob * 100:.2f}%**")
