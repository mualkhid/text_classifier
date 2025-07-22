import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---------- TEXT CLEANING FUNCTION ----------
def clean_text(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# ---------- DATA LOADING ----------
def load_data():
    # Hardcoded dataset (you can replace with CSV if needed)
    data = {
        'text': [
            "I love this place!", "Worst customer service ever.",
            "Amazing coffee and friendly staff.", "I’ll never come back.",
            "Great prices and clean environment.", "Very rude employees.",
            "Absolutely wonderful experience!", "Terrible food quality.",
            "Highly recommend it.", "It was a waste of money."
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    df['cleaned'] = df['text'].apply(clean_text)
    return df

# ---------- VECTORIZATION ----------
def vectorize_text(text_list):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_list)
    return X.toarray(), vectorizer

# ---------- MODEL BUILDING ----------
def build_model(input_size):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(input_size,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ---------- MAIN ----------
def main():
    print("\n📥 Loading and cleaning data...")
    df = load_data()

    print("\n🧠 Vectorizing text...")
    X, vectorizer = vectorize_text(df['cleaned'])
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n🔧 Building and training model...")
    model = build_model(X.shape[1])
    model.fit(X_train, y_train, epochs=20, batch_size=4, verbose=0)

    print("\n✅ Evaluating on test set...")
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Predict on new user input
    while True:
        user_input = input("\n💬 Enter feedback (or type 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            break
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned]).toarray()
        result = model.predict(vector)[0][0]
        sentiment = "Positive ✅" if result > 0.5 else "Negative ❌"
        print(f"🧾 Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
