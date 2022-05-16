import streamlit as st
import tensorflow as tf
import streamlit as st
from PIL import Image
from keras.utils.np_utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import io
from keras.preprocessing.image import load_img
import streamlit as st



image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])

if image_file is not None:
    model = tf.keras.models.load_model('unet_DAPI_1000.hdf5')
    image = Image.open(image_file)
    img_array = np.array(image)
    img_array = img_array[:,:,0]
    image=img_array
    st.write(img_array.shape)
    image2 = cv2.imread('fragment/fragment_image/1 (11).png', 0)
    st.write(image2.shape)
    test_img_other_norm = np.expand_dims(normalize(np.array(image), axis=1),2)
    test_img_other_norm = test_img_other_norm[:,:,0][:,:,None]
    test_img_other_input= np.expand_dims(test_img_other_norm, 0)
    prediction_other = model.predict(test_img_other_input)
    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('External Image')
    st.image(image)
    plt.subplot(232)
    plt.title('Prediction of external Image')
    st.image(prediction_other, cmap='gray')
    plt.show()
    st.image(image, caption='Enter any caption here')
