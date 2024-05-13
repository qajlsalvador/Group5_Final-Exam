import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model


model = load_model('Vege1.h5')


classes = {
    0: 'cabbage', 
    1: 'lettuce', 
    2: 'carrot', 
    3: 'eggplant', 
    4: 'onion'
}

st.title('Vegetable Identifier for Jomel\'s Garden')
st.write("This model predicts the vegetable that can be seen in Jomel's Garden")


class_list = "\n".join([f"- {cls_name}" for cls_name in classes.values()])
st.write(class_list)

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
            predicted_vegetable = classes[pred_class_index]
            st.success(f"Prediction: {predicted_vegetable}")
        else:
            st.warning("Unknown Vegetable")
    except Exception as e:
        st.error("Error processing image. Please upload a valid image.")
