import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dettecao de Chapeu", layout="wide")


def main():

    pages_1 = {
        "Identificador de Chapeu": [
            st.Page("src/pages/conhecimento.py", title="Reconhecimento de Chapeu"),
        ],
    }

    pg = st.navigation(pages_1)
    pg.run()


if __name__ == "__main__":
    main()
