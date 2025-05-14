import streamlit as st


def create_title(title: str):
    st.html(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@700&display=swap');
          .title {{
            font-family: 'Inter', sans-serif;
            font-size: 3.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 3%;
            background: #ffddd2;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 1.2px;
            user-select: none;
          }}
          @media (max-width: 480px) {{
            .title {{
              font-size: 2.5rem;
              margin-bottom: 15%;
            }}
          }}
        </style>
        <div class="title">{title}</div>
        """
    )
