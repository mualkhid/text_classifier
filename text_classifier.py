# Imports
import string
import numpy as np
import pandas as pd
import nltk

# NLTK tools for text processing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tools for converting text and evaluating model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# TensorFlow/Keras for neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Download required NLTK resources
nltk.download('punkt')         # Tokenizer
nltk.download('stopwords')     # Common stopwords
nltk.download('wordnet')       # WordNet Lemmatizer
nltk.download('punkt_tab')  # Ensures NLTK tokenizers work properly


# ------------------------------------------------------------
# STEP 1: Clean and normalize the input text
# ------------------------------------------------------------
def clean_text(text):
    # Lowercase all characters
    text = text.lower().strip()

    # Remove punctuation using translate and string.punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize into words
    tokens = nltk.word_tokenize(text)

    # Remove stopwords (like "the", "is", "and", etc.)
    stop_words = set(stopwords.words('english'))

    # Initialize WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word and remove stopwords
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Return cleaned sentence as a string
    return ' '.join(cleaned)

# ------------------------------------------------------------
# STEP 2: Create a small labeled dataset
# ------------------------------------------------------------
def load_data():
    # Hardcoded examples of positive and negative feedback
    data = {
        'text': [
            "I love this place!",
            "Worst customer service ever.",
            "Amazing coffee and friendly staff.",
            "I’ll never come back.",
            "Great prices and clean environment.",
            "Very rude employees.",
            "Absolutely wonderful experience!",
            "Terrible food quality.",
            "Highly recommend it.",
            "It was a waste of money."
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Clean each feedback message using our function
    df['cleaned'] = df['text'].apply(clean_text)
    return df

# ------------------------------------------------------------
# STEP 3: Convert cleaned text into numeric Bag of Words vectors
# ------------------------------------------------------------
def vectorize_text(text_list):
    # Use CountVectorizer from sklearn to build BoW model
    vectorizer = CountVectorizer()

    # Fit and transform the text list to numeric feature matrix
    X = vectorizer.fit_transform(text_list)

    # Return the matrix and the vectorizer itself
    return X.toarray(), vectorizer

# ------------------------------------------------------------
# STEP 4: Build the Feedforward Neural Network using Keras
# ------------------------------------------------------------
def build_model(input_size):
    model = Sequential()

    # Input layer → Hidden layer with 16 neurons, ReLU activation
    model.add(Dense(16, activation='relu', input_shape=(input_size,)))

    # Output layer → 1 neuron, sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile model with loss and optimizer
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

# ------------------------------------------------------------
# STEP 5: Main flow — Train and use the model
# ------------------------------------------------------------
def main():
    print("\n📥 Loading and cleaning data...")
    df = load_data()  # Get preprocessed dataset

    print("\n🧠 Vectorizing text...")
    X, vectorizer = vectorize_text(df['cleaned'])  # BoW conversion
    y = np.array(df['label'])  # Labels: 1 or 0

    # Split dataset into 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🔧 Building and training model...")
    model = build_model(X.shape[1])  # Input size = vocab size
    model.fit(X_train, y_train, epochs=20, batch_size=4, verbose=0)

    print("\n✅ Evaluating on test set...")
    predictions = (model.predict(X_test) > 0.5).astype("int32")  # Round sigmoid outputs
    print("Accuracy:", accuracy_score(y_test, predictions))

    # --------------------------------------------------------
    # STEP 6: Predict sentiment for user-entered text
    # --------------------------------------------------------
    while True:
        user_input = input("\n💬 Enter feedback (or type 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            break

        # Clean and vectorize user input
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned]).toarray()

        # Predict sentiment
        result = model.predict(vector)[0][0]
        sentiment = "Positive ✅" if result > 0.5 else "Negative ❌"
        print(f"🧾 Sentiment: {sentiment}")

# Run the script
if __name__ == "__main__":
    main()
