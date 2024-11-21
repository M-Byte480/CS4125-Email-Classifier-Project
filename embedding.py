# todo: review unused file

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utilities.configuration.config import Config


def get_tfidf_embd(df: pd.DataFrame, text_column: str = Config.INTERACTION_CONTENT):
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the text data
    embeddings = vectorizer.fit_transform(df[text_column].fillna(""))

    # Return the embeddings as a NumPy array
    return embeddings.toarray()
