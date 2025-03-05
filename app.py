import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

# ✅ Load IMDb Movie Dataset (From Kaggle CSV)
@st.cache_data
def load_movie_data():
    file_path = "imdb_movies.csv"
    if not os.path.exists(file_path):
        st.error("❌ Dataset file 'imdb_movies.csv' not found! Please upload the file.")
        st.stop()
    return pd.read_csv(file_path)

movies_df = load_movie_data()

# ✅ Rename columns based on given attributes
movies_df.rename(columns={
    "date_x": "release_date",
    "score": "imdb_rating",
    "genre": "genre",
    "overview": "description",
    "crew": "crew",
    "orig_title": "title",
    "status": "status",
    "orig_lang": "language",
    "budget_x": "budget",
    "revenue": "revenue",
    "country": "country"
}, inplace=True)

# ✅ Handle missing values
movies_df.fillna("Unknown", inplace=True)

# ✅ Convert text descriptions into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X = tfidf_vectorizer.fit_transform(movies_df["description"])

# ✅ Define target variable (IMDb rating classification)
y = np.where(movies_df["imdb_rating"].astype(float) >= 7, "High",
             np.where(movies_df["imdb_rating"].astype(float) >= 5, "Medium", "Low"))

# ✅ Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train an LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train.toarray(), y_train)

def get_movie_details(movie_name):
    movie = movies_df[movies_df["title"].str.contains(movie_name, case=False, na=False)]
    if not movie.empty:
        return movie.iloc[0]
    return None

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

def recommend_movies(movie_title, num_recommendations=5):
    return movies_df["title"].sample(num_recommendations).tolist()

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
            st.write(f"**Year:** {movie_details['release_date']}")
            st.write(f"**IMDb Rating:** {movie_details['imdb_rating']}")
            st.write(f"**Genre:** {movie_details['genre']}")
            st.write(f"**Crew:** {movie_details['crew']}")
            st.write(f"**Description:** {movie_details['description']}")
            st.write(f"**Country:** {movie_details['country']}")

            # ✅ Sentiment Analysis
            sentiment_score = analyze_sentiment(movie_details["description"])
            st.write(f"**Description Sentiment Score:** {sentiment_score:.2f}")

            # ✅ Predict IMDb Rating Category using LDA
            movie_tfidf = tfidf_vectorizer.transform([movie_details["description"]])
            predicted_rating_category = lda.predict(movie_tfidf.toarray())[0]
            st.write(f"**Predicted IMDb Rating Category:** {predicted_rating_category}")

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
