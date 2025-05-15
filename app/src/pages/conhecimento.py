import streamlit as st
import matplotlib.pyplot as plt
from src.main import deteccacao

st.markdown(
    "<h1 style='text-align: center;'>🧢 Tutorial Identificador de Chapeu</h1>",
    unsafe_allow_html=True,
)

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
# Teórico

## Fundamentação Teórica Aplicada ao Projeto

### Objetivo Geral do Projeto

O presente projeto tem como objetivo o desenvolvimento de um sistema inteligente capaz de **classificar imagens de indivíduos com ou sem chapéu**. A aplicação prática dessa tecnologia visa demonstrar como conceitos de inteligência artificial, aprendizado profundo e visão computacional podem ser aplicados a problemas reais com simplicidade e eficiência. Para tal, foi desenvolvido um modelo de rede neural convolucional (CNN) utilizando as bibliotecas TensorFlow e Keras, além de uma interface interativa desenvolvida com o framework Streamlit.

### Redes Neurais Convolucionais (CNNs)
As CNNs utilizam camadas convolucionais, que aplicam filtros sobre as imagens de entrada para extrair características importantes (como bordas, texturas, padrões), seguidas por camadas de pooling, que reduzem a dimensionalidade dos dados, e camadas densas (fully connected), que tomam decisões de classificação com base nos padrões extraídos.

### Estrutura do Sistema

O sistema foi dividido em três partes principais:

1. **Modelagem e treinamento do modelo de IA**  
   Um modelo CNN foi treinado com um dataset criado manualmente (521 imagens), dividido entre as classes "com chapéu" e "sem chapéu". O modelo atinge cerca de 98% de acurácia no conjunto de validação durante o treinamento e 85,7% em imagens de teste.

2. **Construção da aplicação interativa**  
   A aplicação Streamlit permite ao usuário enviar uma imagem de qualquer pessoa, visualizar a imagem carregada e receber como retorno a predição da rede neural com uma barra de porcentagem indicando a classe atribuída.

3. **Integração e uso prático**  
   O projeto final é capaz de **processar novas imagens em tempo real**, facilitando o consumo de IA sem a necessidade de conhecimento técnico por parte do usuário final. Todo o processo, desde o upload até a exibição do resultado, ocorre de maneira fluida e visualmente amigável.

### Interpretação dos Resultados
Mesmo com uma arquitetura relativamente simples, o modelo apresentou resultados robustos. A acurácia total foi de aproximadamente 85,7% em um conjunto de imagens não vistas durante o treinamento, com precisão perfeita na classe “com chapéu”. Essa performance evidencia a aplicabilidade prática de modelos de IA para tarefas visuais do cotidiano.

Além disso, os resultados são exibidos de forma interativa na aplicação Streamlit, permitindo ao usuário final visualizar a imagem enviada, as probabilidades de classificação e a confiança da predição, com uma interface amigável e visual.
""")


st.markdown(
    """
    # Tutorial
    """
)


st.markdown( """### 1️⃣ Download e Preparação dos Dados""")
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


st.markdown("""### 2️⃣ Pré-processamento das Imagens""")
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


st.markdown("""### 3️⃣ Construção do Modelo CNN""")
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


st.markdown("""### 4️⃣ Compilação e Treinamento""")
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


st.markdown("""### 5️⃣ Configuração Inicial""")
st.markdown(
    """
<p style='margin-left: 20px';>Iremos baixar as bibliotecas essenciais para reconhecer a imagem.</p>

<p style='margin-left: 20px';>Código:</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """
    import os
    import gdown
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    """
)

st.markdown("""### 6️⃣ Função Principal deteccacao()""")
st.markdown(
    """
<p style='margin-left: 20px';>Iremos baixar e carregar o modelo que foi treinado que esta salvo no Google Drive como meu_modelo.h5.</p>

<p style='margin-left: 20px';>Código:</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """
        # Caminho para salvar o modelo localmente
    model_path = "meu_modelo.h5"

    # Baixar o modelo do Google Drive se ainda não estiver salvo
    with st.spinner("Baixando o modelo... Isso pode demorar um pouco."):
        if not os.path.exists(model_path):
            url = "https://drive.google.com/uc?id=11SDM_KTSeNfTZxHoWz-lWsS1AH2F3xON"
            gdown.download(url, model_path, quiet=False, use_cookies=False)

            # Verifica se o arquivo baixado parece válido
            if os.path.getsize(model_path) < 10000:
                st.error("❌ Erro: o arquivo do modelo não foi baixado corretamente.")
                st.stop()

    # Carregar o modelo
    model = load_model(model_path)
    """
)

st.markdown("""### 7️⃣ Pré-processamento de Imagens""")
st.markdown(
    """
<p style='margin-left: 20px';>Redemensionaremos a imagem para a mesma resolucao do treinamento.</p>

<p style='margin-left: 20px';>Código:</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """def carregar_imagem(caminho_img):
    img = image.load_img(caminho_img, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalização
    return np.expand_dims(img_array, axis=0)
        """
)

st.markdown("""### 8️⃣ Função de Predição""")
st.markdown(
    """
<p style='margin-left: 20px';>Iremos fazer a predicao da imagem, retornando a probabilidade dela .</p>

<p style='margin-left: 20px';>Código:</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """
    def predizer(imagem):
    img_array = carregar_imagem(imagem)
    pred = model.predict(img_array)[0][0]
    return 1 - pred, pred  # (prob_com_chapeu, prob_sem_chapeu)"""
)

st.markdown("""### 9️⃣ Upload de Imagem""")
st.markdown(
    """
<p style='margin-left: 20px';>Fazemos o upload da imagem no streamlit .</p>

<p style='margin-left: 20px';>Código:</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """
        uploaded_file = st.file_uploader(
    "📤 Faça upload de uma imagem", 
    type=["jpg", "jpeg", "png"]
)"""
)

st.markdown("""### 🔟 Exibição dos Resultados""")
st.markdown(
    """
<p style='margin-left: 20px';>Mostramos a imagem seleciona e ao lado dela um grafico de barras contendo a porcentagem.</p>

<p style='margin-left: 20px';>Código:</p>
""",
    unsafe_allow_html=True,
)

st.code(
    """
    col1, col2 = st.columns(2)

with col1:
    st.image(img_resized, caption="Imagem enviada", width=350)

with col2:
    # Cria gráfico de barras
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.bar(["Com chapéu", "Sem chapéu"], 
           [prob_com_chapeu, prob_sem_chapeu], 
           color=["#4CAF50", "#F44336"])
    # Configurações visuais...
    st.pyplot(fig)"""
)

st.markdown(""" # Pratico""")
st.markdown(
    """
    <p style='margin-left: 20px ; color: red; font-size:25px' >Agora você pode usar o modelo para classificar novas imagens de chapéus!</p>
""",
    unsafe_allow_html=True,
)

deteccacao()
