
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# Load model and data
model = tf.keras.models.load_model("food_classifier_101.h5") # Or "best_food_classifier_101.h5" if you rename it
recipes_df = pd.read_csv("recipes.csv")
class_names = recipes_df['food'].tolist()

# App UI
st.title(" 🍽️  Food Classifier + Recipe Finder (101 Classes)")
uploaded_file = st.file_uploader("Upload a food image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_food = class_names[predicted_index]
    recipe = recipes_df.loc[recipes_df['food'] == predicted_food, 'recipe'].values[0]

    st.subheader(f" 🍔  Predicted Food: {predicted_food.replace('_',' ').title()}")
    st.markdown(f" 📃  **Recipe:** {recipe}")
