from PIL import Image,ImageOps
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
#st.set_option('deprecation.showfileUploaderEncoding',False)
def load_models(img):
    model = models.load_model('D:\Workspace\python\lab02a.keras')
    image=img.resize((28,28))
    image_array=np.array(image, dtype = np.float32)
    if image_array.shape[-1]==3:
        image_array= tf.image.rgb_to_grayscale(image_array)
    else:
        image_array= np.expand_dims(image_array, axis=-1)
    
    image_array=image_array/255.0
    image_array=np.reshape(image_array,(28,28,1))
    image_array=np.expand_dims(image_array, axis=0)
 
    prediction = model.predict(image_array)
    label = np.argmax(prediction, axis=1)[0]
    return label
def upload_images():
    uploaded_file = st.file_uploader("Choose an Image ..", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded The image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = load_models(image)
        st.write(f"The predicted label is: {label}")
def upload_images():
    uploaded_file = st.file_uploader("Choose an Image ..", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded The image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = load_models(image)
        if label==0:
            st.write("0")
        if label==1:
            st.write("1")
        if label==2:
            st.write("2")
        if label==3:
            st.write("3")
        if label==4:
            st.write("4")
        if label==5:
            st.write("5")
        if label==6:
            st.write("6")
        if label==7:
            st.write("7")
        if label==8:
            st.write("8")
        if label==9:
            st.write("9")

if __name__ =="__main__":
 st.header("MNIST DATA classification")
 st.write("Upload an image")
 
 upload_images()
