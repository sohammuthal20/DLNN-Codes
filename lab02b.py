from PIL import Image,ImageOps
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
#st.set_option('deprecation.showfileUploaderEncoding', False)
def load_models(img):
    model = models.load_model('D:\Workspace\python\Lab02b.keras')
    image=img.resize((32,32))
    image_array=np.array(image)
    #image_array=tf.image.rgb_to_grayscale(image_array)
    image_array=(tf.reshape(image_array,(image_array.shape[0],image_array.shape[0],3)))/255
    image_array=np.array([image_array])
 
    prediction = model.predict(image_array)
    predicted_class= np.argmax(prediction, axis=1)[0]
    return predicted_class

def upload_images():
 uploaded_file = st.file_uploader("Choose an Image ...", type="jpg")
 if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded The image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = load_models(image)
    if label==0:
        st.write("It is an aeroplane")
    if label==1:
        st.write("It is an automobile")
    if label==2:
        st.write("It is a bird")
    if label==3:
        st.write("It is an cat")
    if label==4:
        st.write("It is a deer")
    if label==5:
        st.write("It is a dog")
    if label==6:
        st.write("It is a frog")
    if label==7:
        st.write("It is a horse")
    if label==8:
        st.write("It is a ship")
    if label==9:
        st.write("It is a truck")
 
 
 
if __name__ =="__main__":
 st.header("CIFAR DATA classification(DENSE LAYERS)")
 st.write("Upload an image")
 
 upload_images()
