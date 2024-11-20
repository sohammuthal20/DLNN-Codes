from PIL import Image,ImageOps
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
#st.set_option('deprecation.showfileUploaderEncoding', False)
def load_models(img):
 model = models.load_model('D:\Workspace\python\Lab02.h5')
 image=img.resize((64,64))
 image_array=np.array(image)
 #image_array=tf.image.rgb_to_grayscale(image_array)
 image_array = tf.cast(image_array, tf.float32)/255.0
 image_array=tf.reshape(image_array,(64,64,3))
 image_array=np.array([image_array])
 
 prediction = model.predict(image_array)
 confidence = np.max(prediction)
 return np.argmax(prediction), confidence

def upload_images():
  uploaded_file = st.file_uploader("Choose an Image ...", type="jpg")
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded The image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, confidence = load_models(image)
    confidence_threshold = 0.85
    
    if confidence<confidence_threshold:
      st.write("The image does not belong to any category.")
    else:
      if label==0:
        st.write("It is a CAT")
      elif label==1:
        st.write("It is a Dog")
        
if __name__ =="__main__":
 st.header("CAT AND DOG PREDICTION")
 st.write("Upload an image")
 
 upload_images()