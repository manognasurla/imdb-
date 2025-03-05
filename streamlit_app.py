import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ✅ Load IMDb Movie Dataset (From Kaggle CSV)
@st.cache_data
def load_movie_data():
    return pd.read_csv("imdb_movies.csv")  # Ensure this file is in your repo

movies_df = load_movie_data()

# ✅ Preprocessing: Convert text descriptions into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X = tfidf_vectorizer.fit_transform(movies_df["description"].fillna(""))

y = movies_df["imdb_rating"].fillna(movies_df["imdb_rating"].mean())  # Handle missing ratings

# ✅ Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train a Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

def get_movie_details(movie_name):
    movie = movies_df[movies_df["title"].str.contains(movie_name, case=False, na=False)]
    if not movie.empty:
        return movie.iloc[0]  # Return first match
    return None

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

def recommend_movies(movie_title, num_recommendations=5):
    idx = movies_df[movies_df["title"].str.contains(movie_title, case=False, na=False)].index
    if not idx.empty:
        idx = idx[0]
        cosine_sim = cosine_similarity(X, X)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
        return [movies_df.iloc[i[0]]["title"] for i in sim_scores]
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
            st.write(f"**Title:** {movie_details['title']}")
            st.write(f"**Year:** {movie_details['year']}")
            st.write(f"**IMDb Rating:** {movie_details['imdb_rating']}")
            st.write(f"**Cast:** {movie_details['cast']}")
            st.write(f"**Description:** {movie_details['description']}")

            # ✅ Sentiment Analysis
            sentiment_score = analyze_sentiment(movie_details["description"])
            st.write(f"**Description Sentiment Score:** {sentiment_score:.2f}")

            # ✅ Predict IMDb Rating using Linear Regression
            movie_tfidf = tfidf_vectorizer.transform([movie_details["description"]])
            predicted_rating = regressor.predict(movie_tfidf)[0]
            st.write(f"**Predicted IMDb Rating:** {predicted_rating:.2f}")

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
