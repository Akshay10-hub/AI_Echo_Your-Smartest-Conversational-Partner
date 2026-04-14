import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load DATA & MODEL
df = pd.read_csv("cleaned_data_AI_Echo.csv")
st.title("AI Echo - Sentiment Analysis Dashboard")

required_cols = ['rating', 'cleaned_text', 'date', 'location', 'platform', 'version', 'verified_purchase']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model_dict = pickle.load(open("model.pkl", "rb"))
model = model_dict

# Create Sentiment Column

def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"
df['sentiment'] = df['rating'].apply(get_sentiment)

# User Input Prediction

st.header("Predict Sentiment")
user_input = st.text_area("Enter your review")

# If user clicks Analyze without typing
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        vector = vectorizer.transform([user_input]).toarray()
        prediction = model['model'].predict(vector)[0]
        st.success(f"Predicted Sentiment: {prediction}")

#Visualizations
# Q1: Overall Sentiment
st.header("Overall Sentiment Distribution")
st.bar_chart(df['sentiment'].value_counts())

# Q2: Sentiment vs Rating
st.header("Sentiment vs Rating")
fig, ax = plt.subplots()
sns.countplot(x='rating', hue='sentiment', data=df, ax=ax)
st.pyplot(fig)

# Q3: Word Cloud
st.header("Keywords in Sentiment")
sentiment_type = st.selectbox("Select Sentiment", ["Positive", "Negative"])
text = " ".join(df[df['sentiment'] == sentiment_type]['cleaned_text'])
if text.strip() != "":
    wordcloud = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("No data available for selected sentiment")

# Q4: Sentiment Over Time
st.header("Sentiment Trend Over Time")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df = df[df['date'].notna()]

df['month'] = df['date'].dt.to_period('M').astype(str)
# Convert sentiment to numeric
df['sentiment_num'] = df['sentiment'].map({
    'Positive': 1,
    'Neutral': 0,
    'Negative': -1
})
# Create trend
trend = df.groupby('month')['sentiment_num'].mean().reset_index()
trend = trend.sort_values('month')
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(x='month', y='sentiment_num', data=trend, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Q5: Verified Users
st.header("Verified vs Non-Verified")
fig, ax = plt.subplots()
sns.barplot(x='verified_purchase', y='rating', data=df, ax=ax)
st.pyplot(fig)

# Q6: Review Length vs Sentiment
st.header("Review Length Analysis")
df['review_length'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))
fig, ax = plt.subplots()
sns.boxplot(x='sentiment', y='review_length', data=df, ax=ax)
st.pyplot(fig)

# Q7: Location Analysis
st.header("Sentiment by Location")
top_locations = df['location'].value_counts().head(10).index
df_loc = df[df['location'].isin(top_locations)]
if not df_loc.empty:
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x='location', y='rating', data=df_loc, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("No location data available")

# Q8: Platform Comparison
st.header("Platform Comparison")
fig, ax = plt.subplots()
sns.barplot(x='platform', y='rating', data=df, ax=ax)
st.pyplot(fig)

# Q9: Version Comparison
st.header("Version Performance")
fig, ax = plt.subplots()
sns.barplot(x='version', y='rating', data=df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Q10: Negative Feedback Themes
st.header("Negative Feedback Themes")
neg_text = " ".join(df[df['sentiment'] == "Negative"]['cleaned_text'])

if neg_text.strip() != "":
    wordcloud = WordCloud(background_color='black').generate(neg_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("No negative reviews available")

