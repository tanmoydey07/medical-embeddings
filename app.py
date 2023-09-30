import sys
sys.path.append("..")

import streamlit as st
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Function for cosine similarity
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Configure pandas to display full text
pd.set_option("display.max_colwidth", None)

from top_n import top_n

# Streamlit function
def main():
    # Load data and models
    st.title("Medical Search Engine")
    st.write('Select Model')

    # Streamlit selectbox for model choice
    model_name = st.selectbox("Model", options=['Skipgram', 'Fasttext'])
    st.write('Type your query here')

    # Streamlit text input for user query
    query = st.text_input("Search box")
    column_name = 'Abstract'

    if query:
        # Perform the search and get results
        P, sim = top_n(query, model_name, column_name)
        
        # Create a Plotly table to display results
        fig = go.Figure(data=[go.Table(
            header=dict(values=['ID', 'Title', 'Abstract', 'Publication Date', 'Similarity']),
            cells=dict(values=[list(P['Trial ID'].values), list(P['Title'].values),
                                list(P['Abstract'].values), list(P['Publication date'].values),
                                list(np.around(sim, 4))], align=['center', 'right']))
        ])

        # Update the layout and display the table
        fig.update_layout(height=1700, width=1000, margin=dict(l=0, r=10, t=20, b=20))
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
