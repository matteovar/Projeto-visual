import streamlit as st
import matplotlib.pyplot as plt
from src.main import deteccacao

st.markdown(
    "<h1 style='text-align: center;'>🧢 Tutorial Identificador de Chapeu</h1>",
    unsafe_allow_html=True,
)
# Membros do Projeto
# =============================================
st.markdown(
    """
    <div >
    <h3 style='text-align: left; margin-bottom: 15px;'>👥 Membros do Projeto</h3>
        <p style='margin-left:30px;'> - Matteo Domiciano Varnier - 32158238</p>
        <p style='margin-left:30px;'> - Diogo Lourenzon Hatz - 10402406</p>

    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    
    """
)

st.header("1️⃣ Download e Preparação dos Dados")
st.markdown(
    """
    <p style='margin-left: 20px;'>Vamos baixar e extrair o dataset de imagens para treinamento.</p>

    <p style='margin-left: 20px;'> Código:</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """
import urllib.request
import zipfile
import os


url = "https://drive.usercontent.google.com/download?id=1eaAm9-t_GRBegrVY4E0tQhn-XgADtBk1&export=download..."


urllib.request.urlretrieve(url, 'dataset.zip')


with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')


print(os.listdir('dataset'))
""",
    language="python",
)

st.markdown(
    """
**Explicação:**
- `urllib.request.urlretrieve()` baixa o arquivo ZIP do Google Drive
- `zipfile.ZipFile` extrai os arquivos para a pasta `dataset`
- `os.listdir` mostra o conteúdo da pasta para verificação
"""
)


st.header("2️⃣ Pré-processamento das Imagens")
st.markdown(
    """
    <p style='margin-left: 20px';>Iremos transformar as imagens brutas em um formato padronizado que a rede neural consegue processar, já separando parte dos dados para teste durante o treinamento.</p>

    <p style='margin-left: 20px';>Código:</p>
    """,
    unsafe_allow_html=True,
)

st.code(
    """
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    rescale=1./255,       
    validation_split=0.2  
)


train_data = datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),  
    batch_size=32,
    class_mode='binary',     
    subset='training',       
    seed=123                 
)


val_data = datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation',     
    seed=123
)
""",
    language="python",
)

st.markdown(
    """
**Explicação:**
- `ImageDataGenerator` pré-processa e aumenta os dados automaticamente
- `rescale=1./255` normaliza os valores dos pixels para o intervalo [0, 1]
- `validation_split=0.2` divide automaticamente 20% dos dados para validação
- `target_size=(128, 128)` redimensiona todas as imagens para 128x128 pixels
- `seed=123` garante que a divisão dos dados seja reproduzível
"""
)


st.header("3️⃣ Construção do Modelo CNN")
st.markdown(
    """
       <p style='margin-left: 20px';>Montamos uma estrutura da inteligência artificial que vai aprender a diferenciar imagens com chapéu de sem chapéu.</p>

        <p style='margin-left: 20px';>Código:     </p>
""",
    unsafe_allow_html=True,
)

st.code(
    """
model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
""",
    language="python",
)

st.markdown(
    """
**Explicação das Camadas:**
1. **Conv2D(32, (3,3))**: Primeira camada convolucional com 32 filtros de 3x3 pixels
2. **MaxPooling2D(2,2)**: Reduz a dimensionalidade pela metade
3. **Conv2D(64, (3,3))**: Segunda camada convolucional com 64 filtros
4. **Flatten()**: Transforma os dados em um vetor 1D para as camadas densas
5. **Dense(128)**: Camada totalmente conectada com 128 neurônios
6. **Dense(1, 'sigmoid')**: Camada de saída com ativação sigmoid para classificação binária
"""
)


st.header("4️⃣ Compilação e Treinamento")
st.markdown(
    """
<p style='margin-left: 20px';>Iremos ensinar a rede neural a reconhecer chapéus usando os dados preparados e guarda o conhecimento aprendido.</p>

<p style='margin-left: 20px';>Código:</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """

model.compile(
    optimizer='adam',           
    loss='binary_crossentropy', 
    metrics=['accuracy']        
)


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


model.save('meu_modelo.h5', save_format='h5')
""",
    language="python",
)

st.markdown(
    """
**Explicação:**
- **optimizer='adam'**: Algoritmo de otimização eficiente para ajuste dos pesos
- **loss='binary_crossentropy'**: Função de perda ideal para problemas de classificação binária
- **metrics=['accuracy']**: Acompanha a porcentagem de acertos durante o treinamento
- **epochs=10**: O modelo verá todo o conjunto de dados 10 vezes
- **model.save()**: Salva o modelo treinado no formato HDF5 para uso futuro
"""
)


st.header("5️⃣ Predição em Novas Imagens")
st.markdown(
    """
<p style='margin-left: 20px';> Baixa novas imagens para testar de um zip chamado predict.zinp, extrai o zip para a pasta predict.</p>

<p style='margin-left: 20px';>Código para download das imagens de teste:</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """

url = "https://drive.google.com/uc?export=download&id=11LQr8df8FMAe_V6x8sfumkG-Mdj6_W62"
urllib.request.urlretrieve(url, 'predict.zip')


with zipfile.ZipFile('predict.zip', 'r') as zip_ref:
    zip_ref.extractall('predict')


print(os.listdir('predict'))
""",
    language="python",
)

st.markdown(
    """
<p style='margin-left: 20px';>Classifica cada imagem usando o modelo treinado</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """
import numpy as np
from tensorflow.keras.preprocessing import image

def carregar_imagem(caminho_img):
    
    img = image.load_img(caminho_img, target_size=(128, 128))
    
    img_array = image.img_to_array(img) / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
""",
    language="python",
)

st.markdown(
    """
<p style='margin-left: 20px';>Mostra os resultados visualmente para as imagens com chapeu e sem chapeu</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """

for i in range(len(os.listdir('predict/predict_hat'))):
    path = os.path.join('predict/predict_hat', os.listdir('predict/predict_hat')[i])
    img = carregar_imagem(path)
    predicao = model.predict(img)
    
    
    img_plot = image.load_img(path, target_size=(128, 128))
    plt.imshow(img_plot)
    plt.axis('off')
    plt.title("Sem chapéu" if predicao[0][0] > 0.5 else "Com chapéu")
    plt.show()


for i in range(len(os.listdir('predict/predict_no_hat'))):
    path = os.path.join('predict/predict_no_hat', os.listdir('predict/predict_no_hat')[i])
    img = carregar_imagem(path)
    predicao = model.predict(img)
    
    img_plot = image.load_img(path, target_size=(128, 128))
    plt.imshow(img_plot)
    plt.axis('off')
    plt.title("Sem chapéu" if predicao[0][0] > 0.5 else "Com chapéu")
    plt.show()
""",
    language="python",
)

st.markdown(
    """
**Explicação:**
-  Baixamos um novo conjunto de imagens para teste
- Criamos uma função auxiliar para carregar e pré-processar as imagens no mesmo formato usado no treinamento
- Para cada imagem:
   - Carregamos e pré-processamos
   - Fazemos a predição com o modelo treinado
   - Exibimos a imagem com o resultado da classificação
-    Usamos um threshold de 0.5 para decidir entre "Com chapéu" ou "Sem chapéu"
"""
)


st.markdown(
    """
    <p style='margin-left: 20px ; color: red; font-size:40px' >Agora você pode usar o modelo para classificar novas imagens de chapéus!</p>
""",
    unsafe_allow_html=True,
)

deteccacao()
