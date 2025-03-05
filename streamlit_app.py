import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ✅ Load IMDb Movie Dataset (From Kaggle CSV)
@st.cache_data
def load_movie_data():
    return pd.read_csv("imdb_movies.csv")  # Ensure this file is present

movies_df = load_movie_data()

# ✅ Ensure all necessary columns exist
required_columns = ["date_x", "score", "genre", "overview", "crew", "orig_title", "status",
                    "orig_lang", "budget_x", "revenue", "country"]

for col in required_columns:
    if col not in movies_df.columns:
        st.error(f"❌ Missing column in CSV: {col}")
        st.stop()

# ✅ Preprocessing: Convert movie overview (description) into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X = tfidf_vectorizer.fit_transform(movies_df["overview"].fillna("")).toarray()  # Convert sparse matrix to dense

# ✅ Use 'score' as the target variable (IMDb rating equivalent)
y = movies_df["score"].fillna(movies_df["score"].mean())  # Handle missing ratings

# ✅ Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train a Linear Discriminant Analysis (LDA) model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# ✅ Function to Get Movie Details
def get_movie_details(movie_name):
    movie = movies_df[movies_df["orig_title"].str.contains(movie_name, case=False, na=False)]
    if not movie.empty:
        return movie.iloc[0]  # Return first match
    return None

# ✅ Sentiment Analysis on Movie Overview
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

# ✅ Recommend Similar Movies
def recommend_movies(movie_title, num_recommendations=5):
    idx = movies_df[movies_df["orig_title"].str.contains(movie_title, case=False, na=False)].index
    if not idx.empty:
        idx = idx[0]
        cosine_sim = np.dot(X, X[idx]) / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[idx]))  # Cosine similarity manually
        sim_scores = np.argsort(cosine_sim)[::-1][1:num_recommendations + 1]  # Get top similar movies
        return [movies_df.iloc[i]["orig_title"] for i in sim_scores]
    return []

# ✅ Streamlit UI
st.title("🎬 AI Smart Movie Assistant with IMDb Rating Prediction")
st.write("Search for a movie and get details, reviews, sentiment analysis, recommendations, and predicted IMDb rating!")

# ✅ User Input
movie_name = st.text_input("Enter a movie name", "")

if st.button("Search"):
    if movie_name:
        movie_details = get_movie_details(movie_name)
        if movie_details is not None:
            st.subheader("📌 Movie Details")
            st.write(f"**Title:** {movie_details['orig_title']}")
            st.write(f"**Release Date:** {movie_details['date_x']}")
            st.write(f"**IMDb Score:** {movie_details['score']}")
            st.write(f"**Genre:** {movie_details['genre']}")
            st.write(f"**Crew:** {movie_details['crew']}")
            st.write(f"**Status:** {movie_details['status']}")
            st.write(f"**Language:** {movie_details['orig_lang']}")
            st.write(f"**Budget:** ${movie_details['budget_x']:,}")
            st.write(f"**Revenue:** ${movie_details['revenue']:,}")
            st.write(f"**Country:** {movie_details['country']}")
            st.write(f"**Overview:** {movie_details['overview']}")

            # ✅ Sentiment Analysis
            sentiment_score = analyze_sentiment(movie_details["overview"])
            st.write(f"**Overview Sentiment Score:** {sentiment_score:.2f}")

            # ✅ Predict IMDb Score using LDA
            movie_tfidf = tfidf_vectorizer.transform([movie_details["overview"]]).toarray()
            predicted_score = lda.predict(movie_tfidf)[0]
            st.write(f"**Predicted IMDb Score:** {predicted_score:.2f}")

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

