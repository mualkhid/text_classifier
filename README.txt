Text Classifier - FeedForward AI Prototype

Preprocessing Steps:
- Lowercasing
- Removing punctuation
- Tokenization
- Stopword removal
- Lemmatization

Bag of Words:
- Created using CountVectorizer from scikit-learn
- Converts cleaned text into numerical vectors

Neural Network Design:
- Input Layer: Size = vocabulary size
- Hidden Layer: Dense(16), activation='relu'
- Output Layer: Dense(1), activation='sigmoid'
- Loss: binary_crossentropy
- Optimizer: adam
- Epochs: 20
- Batch size: 4

To test:
- Run the script and enter feedback for prediction.
