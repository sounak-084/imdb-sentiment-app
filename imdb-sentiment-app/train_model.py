import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# 1. Load Dataset (Ensure you have 'IMDB Dataset.csv' from Kaggle in your folder)
print("Loading dataset...")
df = pd.read_csv("IMDB Dataset.csv")

# Take a sample if running locally to speed up execution (e.g., 20,000 rows)
df = df.sample(n=20000, random_state=42).reset_index(drop=True)

# 2. Text Preprocessing Function
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Lowercase
    text = text.lower()
    return text

print("Preprocessing text...")
df['cleaned_review'] = df['review'].apply(clean_text)

# Map sentiments to binary (positive: 1, negative: 0)
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['sentiment_label'], test_size=0.2, random_state=42
)

# 3. Feature Extraction using TF-IDF
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

# 4. Model Training (Logistic Regression)
print("Training model...")
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 5. Save Artifacts
print("Saving model and vectorizer...")
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Training complete! Files saved successfully.")