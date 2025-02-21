import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')
    nltk.download('stopwords')

# Read the scraped data
df = pd.read_csv('data/british-airways_reviews.csv')

# Basic data cleaning
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text
    return ''

# Clean the review content
df['cleaned_content'] = df['content'].apply(clean_text)

# Sentiment Analysis
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df['sentiment'] = df['cleaned_content'].apply(get_sentiment)

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Sentiment Distribution
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='sentiment', bins=30)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')

# 2. Average Rating vs Sentiment
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='sentiment', y='rating')
plt.title('Rating vs Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Rating')

# 3. Word Cloud
plt.subplot(2, 2, 3)
all_words = ' '.join(df['cleaned_content'])
stop_words = set(stopwords.words('english'))
wordcloud = WordCloud(width=800, height=400,
                     background_color='white',
                     stopwords=stop_words,
                     min_font_size=10).generate(all_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Word Cloud of Reviews')

# 4. Rating Distribution
plt.subplot(2, 2, 4)
sns.countplot(data=df, x='rating')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('data/analysis_results.png')
plt.close()

# Calculate key metrics
metrics = {
    'Total Reviews': len(df),
    'Average Rating': df['rating'].mean(),
    'Average Sentiment': df['sentiment'].mean(),
    'Positive Reviews (%)': (df['sentiment'] > 0).mean() * 100,
    'Negative Reviews (%)': (df['sentiment'] < 0).mean() * 100,
    'Neutral Reviews (%)': (df['sentiment'] == 0).mean() * 100,
    'Recommendation Rate (%)': df['recommended'].mean() * 100 if 'recommended' in df.columns else None
}

# Save metrics to CSV
pd.Series(metrics).to_csv('data/analysis_metrics.csv')

print("Analysis complete! Check data/analysis_results.png for visualizations and data/analysis_metrics.csv for metrics.")
