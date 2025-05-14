import os
import gdown
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def deteccacao():
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

    # Função para pré-processar a imagem
    def carregar_imagem(caminho_img):
        img = image.load_img(caminho_img, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    # Função de predição
    def predizer(imagem):
        img_array = carregar_imagem(imagem)
        pred = model.predict(img_array)[0][0]
        return 1 - pred, pred  # com_chapeu, sem_chapeu

    # Upload da imagem
    uploaded_file = st.file_uploader(
        "📤 Faça upload de uma imagem", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Exibir imagem redimensionada
        img = Image.open(uploaded_file)
        img_resized = img.resize((150, 150))

        # Predição
        with st.spinner("🔍 Realizando a predição..."):
            prob_com_chapeu, prob_sem_chapeu = predizer(uploaded_file)

        # Layout com colunas e tamanho reduzido
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_resized, caption="Imagem enviada", width=350)

        with col2:
            categorias = ["Com chapéu", "Sem chapéu"]
            probabilidades = [prob_com_chapeu, prob_sem_chapeu]
            cores = ["#4CAF50", "#F44336"]

            fig, ax = plt.subplots(figsize=(3, 2))  # gráfico pequeno
            ax.bar(categorias, probabilidades, color=cores)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probabilidade", fontsize=8)
            ax.set_title("Predição", fontsize=10)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            for i, v in enumerate(probabilidades):
                ax.text(i, v + 0.05, f"{v*100:.1f}%", ha="center", fontsize=8)

            st.pyplot(fig)
