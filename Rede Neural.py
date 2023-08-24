import tensorflow as tf
import numpy as np

#dados de treinamento
# Velocidades em metros por segundo (m/s)
metros_por_segundo = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)

# Velocidades correspondentes em quilômetros por hora (km/h)
quilometros_por_hora = np.array([0, 18, 36, 54, 72, 90, 108], dtype=float)

# Criação das camadas densas
#Em uma camada densa cada neuronio está conectado com a todos os neuronios da camada anterior
#l1 = tf.keras.layers.Dense(units=1, input_shape=[1]) #entrada
l1 = tf.keras.layers.Dense(units=16, input_shape=[1],activation='linear') #entrada
l2 = tf.keras.layers.Dense(units=16)
l3 = tf.keras.layers.Dense(units=1) #saida

#units: n° de neuronios por camada.
#input_shape: tamanho da entrada.
#activation= 'linear': Função de ativação (função linear), que retorna valores semelhantes as escalas usadas nos dados originais.


# Cria o modelo sequencial e adiciona as camadas densas
#No modelo sequencias as camadas são empilhadas uma após as outras
modelo = tf.keras.Sequential([l1, l2, l3])
#modelo = tf.keras.Sequential([l1])

# Compilação do modelo
modelo.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
#loss='mean_squared_error': Função de perda, nesse caso erro médio quadratico(MSE), função utilizada para minimizar a diferença entre as previsões do 
#modelo e os valores verdadeiros. 
#optimizer=tf.keras.optimizers.Adam: otimizador que será usado durante o treinamento da rede, usado para ajustar os pesos com base no erro calculado na função de perda.
#O (0.1) é a taxa de de aprendizagem.



# Treinamento do modelo
historico = modelo.fit(metros_por_segundo, quilometros_por_hora, epochs=1000, verbose=True)
#metros_por_segundo: São os dados de entrada.
#quilometros_por_hora: dados de saída esperado para o treinamento.
#epochs = 100: número de épocas(iterações) que o modelo será treinado.  


# Realiza a previsão da conversão de 343 m/s (velocidade do som) para km/h
print(modelo.predict([343])) # resultado esperado 1234.8 km/h

#pesos das camadas densas
#print("pesos da primeira camada: {}".format(l1.get_weights()))
#print("pesos da segunda camada: {}".format(l2.get_weights()))
#print("pesos da terceira camada: {}".format(l3.get_weights()))