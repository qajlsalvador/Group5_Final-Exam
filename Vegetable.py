import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model


model = load_model('Vege1.h5')


vegetable_names = {
    0: 'Cabbage', 
    1: 'Eggplant', 
    2: 'Lettuce', 
    3: 'Carrot', 
    4: 'onion'
}


st.title('Jomel\'s Garden Vegetable Identifier')


st.write("This tool identifies vegetables commonly found in Jomel's Garden.")


st.write("The available vegetables are:")
for idx, veg_name in vegetable_names.items():
    st.write(f"- {veg_name}")

uploaded_image = st.file_uploader("Upload an image of a vegetable", type=["jpg", "jpeg", "png"])


if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    try:

        image = Image.open(uploaded_image)
        image = image.resize((224, 224))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        

        pred_probabilities = model.predict(image)
        pred_class_index = np.argmax(pred_probabilities, axis=1)[0]
        

        if pred_class_index in vegetable_names:
            predicted_vegetable = vegetable_names[pred_class_index]
            st.write(f"Prediction: {predicted_vegetable}")
        else:
            st.write("Unknown Vegetable")
    except Exception as e:
        st.write("Error processing image. Please upload a valid image.")
