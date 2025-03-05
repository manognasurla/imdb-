import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ✅ Load IMDb Movie Dataset
@st.cache_data
def load_movie_data():
    try:
        df = pd.read_csv("imdb_movies.csv")  # Ensure file is present
        st.write("✅ Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'imdb_movies.csv' not found!")
        return None

movies_df = load_movie_data()
if movies_df is None:
    st.stop()  # Stop execution if dataset not found

# ✅ Rename Columns to Match Given Attributes
movies_df.rename(columns={
    "date_x": "release_date",
    "score": "imdb_rating",
    "genre": "genres",
    "overview": "description",
    "crew": "crew_info",
    "orig_title": "original_title",
    "status": "status",
    "orig_lang": "original_language",
    "budget_x": "budget",
    "revenue": "revenue",
    "country": "country"
}, inplace=True)

# ✅ Handle Missing Values
movies_df.fillna({"description": "", "imdb_rating": movies_df["imdb_rating"].mean()}, inplace=True)

# ✅ Convert Text to Numerical Features
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X = tfidf_vectorizer.fit_transform(movies_df["description"])  # Transform descriptions
y = movies_df["imdb_rating"]  # IMDb Ratings as target

# ✅ Train-Test Split for Analysis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ✅ Function to Get Movie Details
def get_movie_details(movie_name):
    movie = movies_df[movies_df["original_title"].str.contains(movie_name, case=False, na=False)]
    return movie.iloc[0] if not movie.empty else None

# ✅ Sentiment Analysis on Description
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

# ✅ Recommend Similar Movies
def recommend_movies(movie_title, num_recommendations=5):
    idx = movies_df[movies_df["original_title"].str.contains(movie_title, case=False, na=False)].index
    if not idx.empty:
        idx = idx[0]
        cosine_sim = cosine_similarity(X, X)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
        return [movies_df.iloc[i[0]]["original_title"] for i in sim_scores]
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
            st.write(f"**Title:** {movie_details['original_title']}")
            st.write(f"**Release Date:** {movie_details['release_date']}")
            st.write(f"**IMDb Rating:** {movie_details['imdb_rating']}")
            st.write(f"**Genres:** {movie_details['genres']}")
            st.write(f"**Cast & Crew:** {movie_details['crew_info']}")
            st.write(f"**Status:** {movie_details['status']}")
            st.write(f"**Language:** {movie_details['original_language']}")
            st.write(f"**Budget:** ${movie_details['budget']:,}")
            st.write(f"**Revenue:** ${movie_details['revenue']:,}")
            st.write(f"**Country:** {movie_details['country']}")
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
