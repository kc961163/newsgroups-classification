###############################################################################
# CNN Text Classification on 20 Newsgroups
###############################################################################

import numpy as np
import re
import nltk
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report


###############################################################################
# SECTION 1: Load Vocabulary, Newsgroups, and Labels
###############################################################################
nltk.download('stopwords')  # Ensure NLTK stopwords are available

# 1. Load vocabulary
vocab = []
with open("vocabulary.txt", "r") as f:
    for line in f:
        vocab.append(line.strip())
vocab_size = len(vocab)
print("[INFO] Loaded vocabulary. Size =", vocab_size)

# 2. Load newsgroup names from train.map
newsgroups = [None] * 20
with open("train.map", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        group_name = parts[0]
        label_id = int(parts[1])  # 1-indexed
        newsgroups[label_id - 1] = group_name
print("[INFO] Newsgroups:", newsgroups)

# 3. Load train/test labels
train_labels = []
with open("train.label", "r") as f:
    for line in f:
        train_labels.append(int(line.strip()))
train_labels = np.array(train_labels)
n_train_docs = len(train_labels)
print("[INFO] Number of training documents =", n_train_docs)

test_labels = []
with open("test.label", "r") as f:
    for line in f:
        test_labels.append(int(line.strip()))
test_labels = np.array(test_labels)
n_test_docs = len(test_labels)
print("[INFO] Number of test documents =", n_test_docs)


###############################################################################
# SECTION 2: Reconstruct Text from (doc_id, word_id, count)
###############################################################################
def load_numeric_data_as_dicts(filename, num_docs):
    """
    Returns a list of dictionaries, doc_dicts[doc_id], 
    where each dictionary is {word_id: count} for that document.
    """
    doc_dicts = [{} for _ in range(num_docs)]
    with open(filename, "r") as f:
        for line in f:
            doc_id, word_id, count = map(int, line.strip().split())
            doc_dicts[doc_id - 1][word_id - 1] = count  # store in 0-based
    return doc_dicts

def reconstruct_text(doc_dict, vocabulary):
    """
    Repeats each word in 'vocabulary[word_idx]' 'count' times,
    forming a rough approximation of the original text.
    """
    words = []
    for word_idx, cnt in doc_dict.items():
        # repeat the word 'cnt' times
        words.extend([vocabulary[word_idx]] * cnt)
    text = " ".join(words)
    return text

# Load doc dictionaries
train_doc_dicts = load_numeric_data_as_dicts("train.data", n_train_docs)
test_doc_dicts  = load_numeric_data_as_dicts("test.data",  n_test_docs)

# Reconstruct text
X_train_text = [reconstruct_text(doc, vocab) for doc in train_doc_dicts]
X_test_text  = [reconstruct_text(doc, vocab) for doc in test_doc_dicts]

print("[DEBUG] First training doc (partial):", X_train_text[0][:200], "...")


###############################################################################
# SECTION 3: Preprocessing (Tokenization, Lowercasing, Stopwords, etc.)
###############################################################################
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation/numbers (optional heuristic)
    text = re.sub(r'[^a-z\s]', '', text)
    # 3. Tokenize on whitespace
    tokens = text.split()
    # 4. Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

X_train_tokens = [preprocess_text(doc) for doc in X_train_text]
X_test_tokens  = [preprocess_text(doc) for doc in X_test_text]

print("[DEBUG] Sample tokens from train doc 0:", X_train_tokens[0][:20])


###############################################################################
# SECTION 4: Convert Tokens → Integer Sequences & Pad
###############################################################################
tokenizer = Tokenizer()  # or Tokenizer(num_words=N) if you want to limit vocab
tokenizer.fit_on_texts(X_train_tokens)

X_train_seq = tokenizer.texts_to_sequences(X_train_tokens)
X_test_seq  = tokenizer.texts_to_sequences(X_test_tokens)

# Decide on a max_length for padding
max_length = 400
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=max_length, padding='post', truncating='post')

print("[INFO] Train sequence shape =", X_train_pad.shape)
print("[INFO] Test sequence shape  =", X_test_pad.shape)

# Convert labels from 1-based to 0-based
train_labels_0 = train_labels - 1
test_labels_0  = test_labels - 1

num_classes = len(np.unique(train_labels_0))


###############################################################################
# SECTION 5: Build & Train a CNN with Embedding + Conv1D
###############################################################################
vocab_size_new = len(tokenizer.word_index)  # actual # words found by tokenizer
embedding_dim = 100
epochs = 5
batch_size = 64

model = Sequential()
model.add(Embedding(input_dim=vocab_size_new + 1, 
                    output_dim=embedding_dim,
                    input_length=max_length))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-3),
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train_pad,
    train_labels_0,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)


###############################################################################
# SECTION 6: Evaluate & Compare
###############################################################################
loss, accuracy = model.evaluate(X_test_pad, test_labels_0, verbose=0)
print(f"\n[RESULT] CNN Test Accuracy: {accuracy:.4f}")

y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nCNN Classification Report:")
print(classification_report(test_labels_0, y_pred, target_names=newsgroups))
