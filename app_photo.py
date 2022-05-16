import tensorflow as tf
import streamlit as st
from PIL import Image
from keras.utils.np_utils import normalize
from PIL import Image
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img



image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    model = tf.keras.models.load_model('unet_DAPI_1000.hdf5')
    image = Image.open(image_file)
    img_array = np.array(image)
    img_array = img_array[:,:,0]
    image=img_array
    st.write(img_array.shape)
    test_img_other_norm = np.expand_dims(normalize(np.array(image), axis=1),2)
    test_img_other_norm = test_img_other_norm[:,:,0][:,:,None]
    test_img_other_input= np.expand_dims(test_img_other_norm, 0)
    prediction_other = model.predict(test_img_other_input)
    st.image(image)
    st.image(prediction_other, cmap='gray')
    st.image(image, caption='Enter any caption here')
