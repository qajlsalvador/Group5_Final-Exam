import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model('Vege1.h5')

classes = {0: 'cabbage', 1: 'carrot', 2: 'eggplant', 
           3: 'lettuce', 4: 'onion'}

st.title('Vegetable Identifier')

st.write("This model is created to identify these 5 classes:")
st.write(classes)

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
    try:
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        pred_probabilities = model.predict(image)
        pred_class_index = np.argmax(pred_probabilities, axis=1)[0]
    
        if pred_class_index in classes:
            vege = classes[pred_class_index]
            st.write(f"Predicted Vegetable {vege}")
        else:
            st.write("Unknown Vegetable")
    except Exception as e:
        st.write("Unknown Traffic Sign")
