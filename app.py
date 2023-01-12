import streamlit as st
from PIL import Image
import numpy as np
import math

class Neuron:
  def __init__(self, weights, bias, func):
    self.w = weights
    self.b = bias
    self.f = func
    self.y = 0
  
  def __relu(self, y):
    return 0 if y < 0 else y
  
  def __sigmoid(self, y):
    return 1 / (1 + math.e ** -y)
  
  def __tanh(self, y):
    return math.tanh(y)

  def run(self, input_data):
    x = np.array(input_data)
    w = np.array(self.w)

    if x.size != w.size:
      return 'Los vectores deben tener el mismo tamaño'
    else:
      b = self.b
      self.y = float(np.dot(x, w) + b)
    
      if self.f == 'relu':
        return self.__relu(self.y)
      elif self.f == 'sigmoid':
       return self.__sigmoid(self.y)
      elif self.f == 'tanh':
        return self.__tanh(self.y)
      else:
        return 'La funcion de activación no es correcta'

x_list = []
w_list = []

image = Image.open('neurona.jpg')
st.image(image)

st.title('Simulador de neurona')
neuron_size = st.slider('Cantidad entradas/pesos de la neurona', 1, 5, key='neuron_size')

st.header('Pesos')
columns_w = st.columns(neuron_size)
for i in range(0, neuron_size):
  with columns_w[i]:
    w = st.number_input(f'W{i}', key=f'w{i}')
    w_list.append(np.around(w, 2))

st.header('Entradas')
columns_x = st.columns(neuron_size)
for i in range(0, neuron_size):
  with columns_x[i]:
    x = st.number_input(f'X{i}', key=f'x{i}')
    x_list.append(np.around(x, 2))

st.header('Sesgo')
b = st.number_input('Valor de Bias')
st.header('Función de activación')
activation = st.selectbox(
	'Elige una Función de activación',
	('relu', 'sigmoid', 'tanh')
)

if st.button('Calcular salida', key='button1'):
  n1 = Neuron(weights=w_list, bias=b, func=activation)
  x = x_list
  output = n1.run(input_data=x)
  st.write(f'La salida de la neurona es: {output}')
