import numpy as np
import cv2 #convert images into arrays
import streamlit as st
from PIL import Image, ImageOps

st.write("""
# Cat or Dog identification

This app predicts picture of cat and dog!

""")

st.sidebar.header('upload image')

# Displays the imgae
st.subheader('Input image')

#import tensorflow as tf
#model = tf.keras.models.load_model('model.h5')

from keras.models import load_model
model = load_model('model.h5')


uploaded_file = st.sidebar.file_uploader("Upload image here", type= "jpg")

def import_and_predict(image_data, model):
    
        size = (50,50)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

if uploaded_file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a cat!")
    else:
        st.write("It is a dog!")
    
    st.text("Probability (0: Cat, 1: Dog)")
    st.write(prediction)
