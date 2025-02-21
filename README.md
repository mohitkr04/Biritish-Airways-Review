# ✈️ Airline Review Analysis & Booking Prediction System

## 🚀 Overview
This project leverages **machine learning** to analyze airline customer reviews and predict booking likelihood. It automates data scraping from Skytrax, performs **sentiment analysis**, visualizes insights, and builds a predictive model for booking completion.

## 🔥 Key Features
- **Automated Web Scraping**: Extract airline reviews from [airlinequality.com](https://www.airlinequality.com)
- **Sentiment Analysis**: Analyze customer sentiments from review texts and titles
- **Data Visualization**: Generate insights via **word clouds, rating distributions, and trends**
- **Predictive Modeling**: Train a machine learning model to estimate booking completion likelihood
- **Performance Metrics**: Evaluate model accuracy and feature importance

## 📂 Project Structure
```
📦 airline-analysis/
 ├── 📂 notebooks/          # Jupyter notebooks & scripts
 │   ├── scraping.py        # Web scraper for airline reviews
 │   ├── analysis.py        # Sentiment analysis & visualizations
 │   └── booking_prediction.py  # ML model for booking prediction
 │
 ├── 📂 data/               # Data storage
 │   ├── british-airways_reviews.csv       # Scraped reviews
 │   ├── analysis_results.png              # Visual analysis output
 │   ├── analysis_metrics.csv              # Key metrics
 │   ├── booking_prediction_results.png    # Model insights
 │   ├── booking_metrics.csv               # Performance metrics
 │   └── feature_importance.csv            # Important features
 │
 ├── 📜 README.md           # Project documentation
```

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mohitkr04/British-Airways-Review
cd British-Airways-Review
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 📌 How to Use

### 1️⃣ Scrape Airline Reviews
Run the scraping script to collect review data:
```bash
python notebooks/scraping.py
```

### 2️⃣ Analyze Reviews & Generate Insights
```bash
python notebooks/analysis.py
```

### 3️⃣ Train & Evaluate Prediction Model
```bash
python notebooks/booking_prediction.py
```

## 🏗 Technical Breakdown

### ✈️ Data Collection (scraping.py)
```python
# Example Usage
airline_name = "british-airways"  # Change for different airlines
base_url = f"https://www.airlinequality.com/airline-reviews/{airline_name}"
df = scrape_skytrax_reviews(airline_name, base_url, num_pages=50)
```

### 📊 Sentiment Analysis & Visualization (analysis.py)
```python
# Clean review text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text.lower()  # Convert to lowercase

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, stopwords=stop_words).generate(' '.join(df['cleaned_content']))
```

### 🏆 Booking Prediction Model (booking_prediction.py)
```python
# Feature Engineering
feature_cols = ['review_length', 'month', 'year', 'title_sentiment']
X = processed_df[feature_cols]
y = processed_df['booking_complete']

# Train Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

## 📈 Performance Metrics
**Booking Prediction Model Results:**
```
Mean CV Score: 0.852
CV Score Std: 0.044
Test Set Accuracy: 92%
Number of Features: 4
Top Feature: title_sentiment
Top Feature Importance: 0.5134
```

## 📊 Insights from Analysis
- **Sentiment Distribution**: Identifies positive vs. negative reviews
- **Review Trends**: Detects seasonal variations
- **Top Factors Influencing Bookings**: Helps airlines enhance customer experience

## 🔄 Data Processing Workflow
### 📌 Text Processing
✔ Remove special characters & digits  
✔ Convert to lowercase  
✔ Remove stopwords  
✔ Tokenization & sentiment scoring  

### 🔍 Feature Engineering
✔ Review length calculation  
✔ Extract date features  
✔ Compute sentiment scores  
✔ Generate target variable (booking completion)  

## 📦 Dependencies
Ensure you have the following installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `textblob`
- `nltk`
- `matplotlib`
- `seaborn`
- `wordcloud`

## 🛡 Best Practices
### ✅ Data Collection
- Schedule regular scraping
- Validate data to avoid missing values
- Implement error handling for network issues

### ✅ Analysis & Modeling
- Apply consistent preprocessing
- Conduct cross-validation
- Retrain the model periodically

## 🔧 Troubleshooting
**Scraping Issues**
- Ensure internet connectivity
- Verify the target URL
- Handle website rate limits

**Analysis Issues**
- Check for missing values
- Validate date formats

**Model Issues**
- Ensure features are properly extracted
- Check for data leakage

## 🤝 Contributing
1. **Fork** the repository  
2. **Create** a new feature branch (`git checkout -b feature-name`)  
3. **Commit** your changes (`git commit -m "Added new feature"`)  
4. **Push** to the branch (`git push origin feature-name`)  
5. **Submit a Pull Request**  

## 📩 Contact & Support
For questions, feature requests, or bug reports, please **open an issue** in the repository.

## 🙏 Acknowledgments
- **Skytrax** for providing review data
- Open-source contributors
- The AI & data science community

---
⚠️ **Disclaimer:** This project is for **educational purposes only**. Data usage should comply with applicable terms and conditions.  
