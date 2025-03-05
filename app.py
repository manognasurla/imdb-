import streamlit as st
import pandas as pd
import numpy as np
import os
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ✅ Load IMDb Movie Dataset
@st.cache_data
def load_movie_data():
    file_path = "imdb_movies.csv"
    if not os.path.exists(file_path):
        st.error("Dataset not found! Ensure 'imdb_movies.csv' is in the correct directory.")
        return None
    return pd.read_csv(file_path)

movies_df = load_movie_data()
if movies_df is None:
    st.stop()

# ✅ Ensure correct column names
required_columns = {"date_x", "score", "genre", "overview", "crew", "orig_title", "status", "orig_lang", "budget_x", "revenue", "country"}
if not required_columns.issubset(movies_df.columns):
    st.error("Dataset is missing required columns! Check the CSV file format.")
    st.stop()

# ✅ Preprocessing
movies_df = movies_df.fillna("")  # Fill missing values

# ✅ Text Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(movies_df["overview"])
y = movies_df["score"].astype(float)

# ✅ Apply LDA Model
lda = LinearDiscriminantAnalysis()
try:
    X_dense = X.toarray()  # LDA requires dense input
    lda.fit(X_dense, y)
except Exception as e:
    st.error(f"LDA Model Error: {str(e)}")
    st.stop()

# ✅ Movie Details Retrieval
def get_movie_details(movie_name):
    movie = movies_df[movies_df["orig_title"].str.contains(movie_name, case=False, na=False)]
    if not movie.empty:
        return movie.iloc[0]
    return None

# ✅ Sentiment Analysis
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

# ✅ Recommendations
def recommend_movies(movie_title, num_recommendations=5):
    idx = movies_df[movies_df["orig_title"].str.contains(movie_title, case=False, na=False)].index
    if not idx.empty:
        idx = idx[0]
        cosine_sim = cosine_similarity(X, X)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
        return [movies_df.iloc[i[0]]["orig_title"] for i in sim_scores]
    return []

# ✅ Streamlit UI
st.title("🎬 AI Smart Movie Assistant with LDA Analysis")
st.write("Search for a movie and get details, sentiment analysis, recommendations, and LDA-based score prediction!")

# ✅ User Input
movie_name = st.text_input("Enter a movie name", "")

if st.button("Search"):
    if movie_name:
        movie_details = get_movie_details(movie_name)
        if movie_details is not None:
            st.subheader("📌 Movie Details")
            st.write(f"**Title:** {movie_details['orig_title']}")
            st.write(f"**Date:** {movie_details['date_x']}")
            st.write(f"**Score:** {movie_details['score']}")
            st.write(f"**Genre:** {movie_details['genre']}")
            st.write(f"**Crew:** {movie_details['crew']}")
            st.write(f"**Overview:** {movie_details['overview']}")
            
            # ✅ Sentiment Analysis
            sentiment_score = analyze_sentiment(movie_details["overview"])
            st.write(f"**Overview Sentiment Score:** {sentiment_score:.2f}")
            
            # ✅ Predict Score using LDA
            movie_tfidf = vectorizer.transform([movie_details["overview"]]).toarray()
            predicted_score = lda.predict(movie_tfidf)[0]
            st.write(f"**Predicted Score:** {predicted_score:.2f}")
            
            # ✅ Recommendations
            similar_movies = recommend_movies(movie_name)
            if similar_movies:
                st.subheader("🎥 Similar Movies")
                st.write(", ".join(similar_movies))
            else:
                st.write("❌ No similar movies found.")
        else:
            st.error("❌ Movie not found! Showing similar movies...")
            similar_movies = recommend_movies(movie_name)
            if similar_movies:
                st.write("🎥 Recommended Similar Movies:")
                st.write(", ".join(similar_movies))
            else:
                st.write("❌ No recommendations available.")
    else:
        st.warning("⚠️ Please enter a movie name.")
