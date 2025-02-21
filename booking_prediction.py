import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Read the scraped reviews data
df = pd.read_csv('data/british-airways_reviews.csv')

# Data preprocessing
def prepare_data(df):
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Convert rating to binary target (positive/negative experience)
    # Ratings > 5 are considered positive
    data['booking_complete'] = (data['rating'] > 5).astype(int)
    
    # Extract features from review text
    data['review_length'] = data['content'].str.len()
    
    # Convert date to datetime and extract features
    data['date'] = pd.to_datetime(data['date'], format='%dth %B %Y', errors='coerce')
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    
    # Create sentiment score using review title
    from textblob import TextBlob
    data['title_sentiment'] = data['title'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0
    )
    
    # Handle missing values
    data = data.fillna(0)
    
    return data

# Prepare the data
processed_df = prepare_data(df)

# Define features and target
feature_cols = ['review_length', 'month', 'year', 'title_sentiment']

X = processed_df[feature_cols]
y = processed_df['booking_complete']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Generate evaluation metrics
classification_metrics = classification_report(y_test, y_pred)

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Feature Importance Plot
plt.subplot(2, 2, 1)
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance Score')

# 2. Confusion Matrix
plt.subplot(2, 2, 2)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 3. Cross-validation Scores
plt.subplot(2, 2, 3)
plt.boxplot(cv_scores)
plt.title('Cross-validation Scores Distribution')
plt.ylabel('Accuracy')

# 4. ROC Curve
from sklearn.metrics import roc_curve, auc
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.subplot(2, 2, 4)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.tight_layout()
plt.savefig('data/booking_prediction_results.png')
plt.close()

# Save model and metrics
joblib.dump(rf_model, 'data/booking_model.joblib')

# Save metrics to CSV
metrics = {
    'Mean CV Score': cv_scores.mean(),
    'CV Score Std': cv_scores.std(),
    'Test Set Accuracy': rf_model.score(X_test, y_test),
    'Number of Features': len(feature_cols),
    'Top Feature': feature_importance.iloc[0]['feature'],
    'Top Feature Importance': feature_importance.iloc[0]['importance']
}

pd.Series(metrics).to_csv('data/booking_metrics.csv')

# Save feature importance to CSV
feature_importance.to_csv('data/feature_importance.csv', index=False)

print("Analysis complete! Check the data folder for results.")
print("\nModel Performance Summary:")
print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
print("\nClassification Report:")
print(classification_metrics) 