import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
st.title('Anagnórisis de numeros escritos a mano')
st.subheader("Dibuja un numero en el panel y luego dale al boton que dice 'tunometecabrasaramambiche'")
st.text("Lo que va a hacer es que un modelo entrenado te va a decir que número escribiste")

# Add canvas component
# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
color = st.color_picker("Selecciona un color a tu gusto para lo que dibujes, o el stroke", "#FFFFFF")
color_bg = st.color_picker("Ahora selecciona el color del fondo del tablero", "#000000")
stroke_color = '#FFFFFF' # Set background color to white
bg_color = '#000000'

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=color,
    background_color=color_bg,
    height=200,
    width=200,
    key="canvas",
)

# Add "Predict Now" button
if st.button('tunometecabrasaramambiche', type="primary"):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header('El número que dibujaste essss : ' + str(res))
    else:
        st.header('Parece que no has dibujado nada...')

# Add sidebar
st.image("")
