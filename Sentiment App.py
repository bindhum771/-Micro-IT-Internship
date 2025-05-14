import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    'text': [
        "Amazing product!", "Worst experience ever.", "It’s fine, nothing special.",
        "Superb customer service!", "Totally disappointed.", "It's okay I guess.",
        "Very happy with the results.", "Not what I expected.", "Average at best.",
        "Fast delivery and great support.", "It doesn't meet the expectations.", "Mediocre experience."
    ],
    'label': [
        'positive', 'negative', 'neutral',
        'positive', 'negative', 'neutral',
        'positive', 'negative', 'neutral',
        'positive', 'negative', 'neutral'
    ]
}

df = pd.DataFrame(data)

# ML pipeline
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)

# Streamlit App
st.set_page_config(page_title="🔍 Real-Time Sentiment Analysis", page_icon="💬", layout="centered")
st.title("🔍 Real-Time Sentiment Analysis Dashboard")
st.markdown("Get real-time insights from customer feedback like big brands do.")

# Sidebar input mode
input_mode = st.sidebar.radio("Choose Input Mode:", ["Single Text Input", "Bulk CSV Upload"])

# Emoji map
emoji_map = {
    "positive": "🟢 Positive",
    "negative": "🔴 Negative",
    "neutral": "🟡 Neutral"
}

# Single Text Input
if input_mode == "Single Text Input":
    user_input = st.text_area("✍️ Enter a review, tweet, or comment:")
    if st.button("🔍 Analyze"):
        if user_input.strip():
            pred = pipeline.predict([user_input])[0]
            prob = pipeline.predict_proba([user_input])[0]
            st.markdown(f"### Sentiment: **{emoji_map[pred]}**")
            st.progress(int(max(prob)*100))

            # Probability chart
            prob_df = pd.DataFrame({
                'Sentiment': pipeline.classes_,
                'Confidence': prob
            })

            fig, ax = plt.subplots()
            sns.barplot(x='Sentiment', y='Confidence', data=prob_df, ax=ax, palette='pastel')
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)
        else:
            st.warning("Please enter some text.")

# Bulk CSV Upload
else:
    uploaded_file = st.file_uploader("📂 Upload a CSV file with a 'text' column", type=['csv'])
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        if 'text' in df_upload.columns:
            df_upload['prediction'] = pipeline.predict(df_upload['text'])
            st.dataframe(df_upload)

            # Sentiment counts
            sentiment_counts = df_upload['prediction'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']

            # Chart
            st.subheader("📊 Sentiment Distribution")
            fig2, ax2 = plt.subplots()
            sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, palette='Set2', ax=ax2)
            st.pyplot(fig2)

            # Download results
            st.download_button("📥 Download Predictions", df_upload.to_csv(index=False), "predictions.csv", "text/csv")
        else:
            st.error("CSV must contain a 'text' column.")

