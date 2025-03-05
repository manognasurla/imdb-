import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ✅ Load IMDb Movie Dataset
@st.cache_data
def load_movie_data():
    return pd.read_csv("imdb_movies.csv")  # Ensure this file exists

movies_df = load_movie_data()

# ✅ Debug: Check Available Columns
st.write("🔍 Checking dataset columns:", movies_df.columns.tolist())

# ✅ Preprocessing: Convert text descriptions into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X = tfidf_vectorizer.fit_transform(movies_df["overview"].fillna(""))

# ✅ Convert "score" into categories for LDA classification
movies_df["score_category"] = pd.cut(
    movies_df["score"], bins=[0, 5, 7, 10], labels=["Low", "Medium", "High"]
)

# ✅ Use "score_category" for classification
y = movies_df["score_category"].astype(str)

# ✅ Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)

# ✅ Train a Linear Discriminant Analysis model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

def get_movie_details(movie_name):
    movie = movies_df[movies_df["orig_title"].str.contains(movie_name, case=False, na=False)]
    if not movie.empty:
        return movie.iloc[0]  # Return first match
    return None

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

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
st.title("🎬 AI Smart Movie Assistant with IMDb Rating Prediction")
st.write("Search for a movie and get details, reviews, sentiment analysis, recommendations, and predicted IMDb rating category!")

# ✅ User Input
movie_name = st.text_input("Enter a movie name", "")

if st.button("Search"):
    if movie_name:
        movie_details = get_movie_details(movie_name)
        if movie_details is not None:
            st.subheader("📌 Movie Details")
            st.write(f"**Title:** {movie_details['orig_title']}")
            st.write(f"**Year:** {movie_details['date_x']}")
            st.write(f"**IMDb Score:** {movie_details['score']}")
            st.write(f"**Genre:** {movie_details['genre']}")
            st.write(f"**Crew:** {movie_details['crew']}")
            st.write(f"**Description:** {movie_details['overview']}")
            st.write(f"**Budget:** ${movie_details['budget_x']}")
            st.write(f"**Revenue:** ${movie_details['revenue']}")
            st.write(f"**Country:** {movie_details['country']}")

            # ✅ Sentiment Analysis
            sentiment_score = analyze_sentiment(movie_details["overview"])
            st.write(f"**Overview Sentiment Score:** {sentiment_score:.2f}")

            # ✅ Predict IMDb Rating Category using LDA
            movie_tfidf = tfidf_vectorizer.transform([movie_details["overview"]])
            predicted_category = lda.predict(movie_tfidf.toarray())[0]
            st.write(f"**Predicted IMDb Rating Category:** {predicted_category}")

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
