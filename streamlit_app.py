import pandas as pd
import numpy as np
import streamlit as st
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st

try:
    import sklearn
    st.write(f"✅ scikit-learn Version: {sklearn.__version__}")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    st.write("✅ LinearDiscriminantAnalysis is working!")
except ImportError as e:
    st.error(f"❌ ImportError: {e}")
except ModuleNotFoundError as e:
    st.error(f"❌ ModuleNotFoundError: {e}")

# Load IMDb dataset
try:
    df = pd.read_csv("imdb_movies.csv")
except FileNotFoundError:
    st.error("Dataset 'imdb_movies.csv' not found. Please upload the correct file.")
    st.stop()

# Preprocess data
scaler = StandardScaler()
numerical_features = ["budget_x", "revenue"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Categorize 'score' into 'low', 'medium', 'high'
df['score_category'] = pd.cut(df['score'], bins=[-float('inf'), 5, 7, float('inf')], labels=['low', 'medium', 'high'])

# Select features and target
x = df[["budget_x", "revenue", 'names', 'date_x', 'country']]
y = df["score_category"]

# Encode categorical variables
label_encoders = {}
for col in x.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col].astype(str))
    label_encoders[col] = le

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Train LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

# Streamlit UI
st.title("🎬 IMDb Movie Score Prediction")
st.write("Enter movie details to predict its IMDb score category (Low, Medium, High).")

# User input fields
budget = st.number_input("💰 Budget ($)", min_value=0.0, step=100000.0)
revenue = st.number_input("📈 Revenue ($)", min_value=0.0, step=100000.0)
name = st.text_input("🎞 Movie Name")
date = st.text_input("📅 Release Date (YYYY-MM-DD)")
country = st.text_input("🌍 Country")

# Convert inputs using label encoders if necessary
def encode_value(column, value):
    if value in label_encoders[column].classes_:
        return label_encoders[column].transform([value])[0]
    else:
        return -1  # Assign an unknown value indicator

if st.button("🔍 Predict Score Category"):
    if not name or not date or not country:
        st.warning("Please fill in all fields before predicting.")
    else:
        name_encoded = encode_value('names', name)
        date_encoded = encode_value('date_x', date)
        country_encoded = encode_value('country', country)
        
        # Prepare input data for prediction
        user_input = np.array([[budget, revenue, name_encoded, date_encoded, country_encoded]])
        user_input[:, :2] = scaler.transform(user_input[:, :2])  # Scale numerical features
        
        # Make prediction
        prediction = lda.predict(user_input)[0]
        st.success(f"🎯 Predicted IMDb Score Category: **{prediction}**")
