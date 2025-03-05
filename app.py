import streamlit as st
import pandas as pd
import numpy as np
import os
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ✅ Load IMDb Movie Dataset (With File Path Fix)
@st.cache_data
def load_movie_data():
    file_path = os.path.join(os.path.dirname(__file__), "imdb_movies.csv")
    if not os.path.exists(file_path):
        st.error("❌ Error: 'imdb_movies.csv' not found! Please upload the correct dataset.")
        st.stop()  # Stop execution if the file is missing
    return pd.read_csv(file_path)

movies_df = load_movie_data()

# ✅ Rename columns based on provided attributes
movies_df.rename(columns={
    "date_x": "release_date",
    "score": "imdb_rating",
    "genre": "genre",
    "overview": "description",
    "crew": "crew",
    "orig_title": "title",
    "status": "status",
    "orig_lang": "original_language",
    "budget_x": "budget",
    "revenue": "revenue",
    "country": "country"
}, inplace=True)

# ✅ Handle missing values
movies_df["description"].fillna("", inplace=True)
movies_df["imdb_rating"].fillna(movies_df["imdb_rating"].mean(), inplace=True)

# ✅ Convert IMDb rating into categories for classification
movies_df["rating_category"] = pd.cut(
    movies_df["imdb_rating"], bins=[0, 5, 7, 10], labels=["Low", "Medium", "High"]
)

# ✅ Text Processing: Convert movie descriptions into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf_vectorizer.fit_transform(movies_df["description"])

# ✅ Encode labels for LDA classification
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(movies_df["rating_category"])

# ✅ Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf.toarray(), y_encoded, test_size=0.2, random_state=42)

# ✅ Train Linear Discriminant Analysis (LDA) model
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)

# ✅ Function to Get Movie Details
def get_movie_details(movie_name):
    movie = movies_df[movies_df["title"].str.contains(movie_name, case=False, na=False)]
    return movie.iloc[0] if not movie.empty else None

# ✅ Sentiment Analysis on Movie Description
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

# ✅ Recommend Similar Movies
def recommend_movies(movie_title, num_recommendations=5):
    idx = movies_df[movies_df["title"].str.contains(movie_title, case=False, na=False)].index
    if not idx.empty:
        idx = idx[0]
        cosine_sim = cosine_similarity(X_tfidf, X_tfidf)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
        return [movies_df.iloc[i[0]]["title"] for i in sim_scores]
    return []

# ✅ Streamlit UI
st.title("🎬 AI Smart Movie Assistant with IMDb Rating Classification")
st.write("Search for a movie and get details, reviews, sentiment analysis, recommendations, and predicted IMDb rating category!")

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
            st.write(f"**Cast:** {movie_details['crew']}")
            st.write(f"**Description:** {movie_details['description']}")

            # ✅ Sentiment Analysis
            sentiment_score = analyze_sentiment(movie_details["description"])
            st.write(f"**Description Sentiment Score:** {sentiment_score:.2f}")

            # ✅ Predict IMDb Rating Category using LDA
            movie_tfidf = tfidf_vectorizer.transform([movie_details["description"]])
            predicted_category = lda_model.predict(movie_tfidf.toarray())[0]
            predicted_label = label_encoder.inverse_transform([predicted_category])[0]
            st.write(f"**Predicted IMDb Rating Category:** {predicted_label}")

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

