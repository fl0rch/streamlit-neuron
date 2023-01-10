import streamlit as st
from PIL import Image

image = Image.open('neurona.jpg')
st.image(image)

st.title('¡Hola neurona!')

tab1, tab2, tab3 = st.tabs(['Una entrada', 'Dos entradas', 'Tres entradas y sesgo'])

with tab1:
  st.header('Una neurona con una entrada y un peso')
  
  w = st.slider('Peso W', 0.0, 5.0, key='w')
  x = st.number_input('Entrada X', key='x')

  if st.button('Calcular salida', key='button1'):
    y = x * w
    st.write(f'La salida de la neurona es: {y}')

with tab2:
  st.header("Una neurona con dos entrada y dos pesos")

  col1, col2 = st.columns(2)
  with col1:
    w0 = st.slider('Peso W0', 0.0, 5.0, key='w0')
    x0 = st.number_input('Entrada X0', key='x0')
  
  with col2:
    w1 = st.slider('Peso W1', 0.0, 5.0, key='w1')
    x1 = st.number_input('Entrada X1', key='x1')

  if st.button('Calcular salida', key='button2'):
    y = (x0 * w0) + (x1 * w1)
    st.write(f'La salida de la neurona es: {y}')

with tab3:
  st.header("Una neurona con tres entrada, tres pesos y sesgo")

  col1, col2, col3 = st.columns(3)
  with col1:
    w0 = st.slider('Peso W0', 0.0, 5.0, key='w00')
    x0 = st.number_input('Entrada X0', key='x00')
  
  with col2:
    w1 = st.slider('Peso W1', 0.0, 5.0, key='w10')
    x1 = st.number_input('Entrada X1', key='x10')
  
  with col3:
    w2 = st.slider('Peso W2', 0.0, 5.0, key='w20')
    x2 = st.number_input('Entrada X2', key='x20')
  
  b = st.number_input('Sesgo')

  if st.button('Calcular salida', key='button 3'):
    y = (x0 * w0) + (x1 * w1) + (x2 * w2) + b
    st.write(f'La salida de la neurona es: {y}')
