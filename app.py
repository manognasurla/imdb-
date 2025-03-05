import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ✅ Load IMDb Movie Dataset
@st.cache_data

def load_movie_data():
    try:
        df = pd.read_csv("imdb_movies.csv")  # Ensure this file exists
        return df
    except FileNotFoundError:
        st.error("❌ Error: Dataset file 'imdb_movies.csv' not found!")
        return None

movies_df = load_movie_data()
if movies_df is None:
    st.stop()

# ✅ Preprocess dataset (only keeping necessary columns)
columns_needed = ["date_x", "score", "genre", "overview", "crew", "orig_title", "status", "orig_lang", "budget_x", "revenue", "country"]
movies_df = movies_df[columns_needed].dropna()

# ✅ Convert categorical columns to numerical
movies_df["genre"] = movies_df["genre"].astype("category").cat.codes
movies_df["crew"] = movies_df["crew"].astype("category").cat.codes
movies_df["orig_lang"] = movies_df["orig_lang"].astype("category").cat.codes
movies_df["status"] = movies_df["status"].astype("category").cat.codes
movies_df["country"] = movies_df["country"].astype("category").cat.codes

# ✅ Text feature transformation
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
X_text = tfidf_vectorizer.fit_transform(movies_df["overview"].fillna(""))

# ✅ Combine text features with numerical data
X_numeric = movies_df.drop(columns=["overview", "score"])
X = np.hstack((X_numeric.values, X_text.toarray()))
y = movies_df["score"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train the Linear Discriminant Analysis model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# ✅ Function to Get Movie Details
def get_movie_details(movie_name):
    movie = movies_df[movies_df["orig_title"].str.contains(movie_name, case=False, na=False)]
    if not movie.empty:
        return movie.iloc[0]  # Return first match
    return None

# ✅ Sentiment Analysis
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

# ✅ Streamlit UI
st.title("🎬 AI Smart Movie Assistant with LDA")
st.write("Search for a movie and get details, sentiment analysis, and predicted scores!")

# ✅ User Input
movie_name = st.text_input("Enter a movie name", "")

if st.button("Search"):
    if movie_name:
        movie_details = get_movie_details(movie_name)

        if movie_details is not None:
            st.subheader("📌 Movie Details")
            st.write(f"**Title:** {movie_details['orig_title']}")
            st.write(f"**Genre Code:** {movie_details['genre']}")
            st.write(f"**Status:** {movie_details['status']}")
            st.write(f"**Language Code:** {movie_details['orig_lang']}")
            st.write(f"**Budget:** {movie_details['budget_x']}")
            st.write(f"**Revenue:** {movie_details['revenue']}")
            st.write(f"**Country Code:** {movie_details['country']}")
            st.write(f"**Overview:** {movie_details['overview']}")

            # ✅ Sentiment Analysis
            sentiment_score = analyze_sentiment(movie_details["overview"])
            st.write(f"**Overview Sentiment Score:** {sentiment_score:.2f}")

            # ✅ Predict movie score using LDA
            movie_features = np.hstack(([movie_details[col] for col in X_numeric.columns], tfidf_vectorizer.transform([movie_details["overview"]]).toarray()))
            predicted_score = lda.predict([movie_features])[0]
            st.write(f"**Predicted Score:** {predicted_score:.2f}")

        else:
            st.error("❌ Movie not found! Please try another title.")

    else:
        st.warning("⚠️ Please enter a movie name.")
