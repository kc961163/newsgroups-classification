# Machine Learning Project: Newsgroups Classification using Generative and Discriminative Models

## Introduction

*This project focuses on classifying newsgroup posts using two types of models: a generative model (Naive Bayes) and a discriminative model (Logistic Regression and Convoluted Neural Network). We use the 20 Newsgroups dataset to train, evaluate, and compare the performance of these models.*

## How to Download the Data

The dataset can be obtained from the following URL:  
[20 Newsgroups Dataset](http://people.csail.mit.edu/jrennie/20Newsgroups/)

We recommend downloading the **"bydate"** version because it provides:
- A predefined train/test split,
- Duplicates removed,
- And some newsgroup-identifying headers stripped out.

### Data Versions Available

1. **Numeric (Matlab/Octave) Version:**
   - **File:** `20news-bydate-matlab.tgz`  
   - **Description:**  
     This processed version is designed for ease of use in Matlab/Octave as a sparse matrix. It contains six files:
     - `train.data` – Document-word count data for the training set (formatted as "docIdx wordIdx count").
     - `train.label` – A list of label IDs for the training documents.
     - `train.map` – A mapping from label IDs to newsgroup names.
     - `test.data` – Document-word count data for the test set.
     - `test.label` – A list of label IDs for the test documents.
     - `test.map` – A mapping from label IDs to newsgroup names.
   - **Additional File:**  
     - `vocabulary.txt` – Contains the list of words with line numbers corresponding to word indices (e.g., word on the first line is word #1, second line is word #2, etc.).
   - **Note:**  
     These files were produced using the scripts `lexData.sh` and `rainbow2matlab.py`.

2. **Raw Text Files:**
   - **Description:**  
     If needed, you can also download the raw version of the dataset (often provided as directories named by newsgroup) which contains individual text files for each post. This raw data can be useful for custom preprocessing or alternative parsing methods.

Use the numeric (Matlab/Octave) version for quick experiments with sparse matrices and the raw text version if you need to perform your own text processing or experiments with deep learning models.

## Data Description

The dataset is organized into six primary files (or bundles) that provide a numeric, machine-friendly representation of the text data:

1. **vocabulary.txt**  
    - **Description:**  
      A list of words that may appear in the documents. Each line corresponds to a word, and the line number is the word's unique ID.  
    - **Example:**  
      - Line 1: `archive` (wordId 1)  
      - Line 2: `name` (wordId 2)

2. **train.map**  
    - **Description:**  
      A mapping of newsgroup names to their label IDs. Each line contains a newsgroup name followed by its corresponding ID.  
    - **Example:**  
      ```
      alt.atheism 1
      comp.graphics 2
      comp.os.ms-windows.misc 3
      ...
      ```
    - **Note:**  
      Since we do not have a separate **newsgrouplabels.txt** file, we use **train.map** (and optionally **test.map** if provided) to obtain the newsgroup names.

3. **train.label**  
    - **Description:**  
      Each line in this file contains the label (as a number) for a single training document. The document ID corresponds to the line number.  
    - **Example:**  
      - Line 1: `1` indicates that the first document belongs to the newsgroup with label ID 1 (i.e., *alt.atheism*).

4. **test.label**  
    - **Description:**  
      Similar to **train.label**, this file contains the labels for the test documents, with the document ID corresponding to the line number.

5. **train.data**  
    - **Description:**  
      This file specifies the counts of words in each training document. Each line is formatted as:  
      ```
      docId wordId count
      ```
      - **docId:** The ID of the document (from **train.label**).  
      - **wordId:** The ID of the word (from **vocabulary.txt**).  
      - **count:** The number of times the word appears in the document.  
    - **Important:**  
      Any word/document pair not present in the file is assumed to have a count of 0.

6. **test.data**  
    - **Description:**  
      The same format as **train.data**, but it contains the word counts for the test documents.

---

## Getting Started

1. **Clone the Repository**  
    Clone the repository and navigate to the project directory:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Set Up the Conda Environment**  

    Create and activate a new conda environment with Python 3.9:
    ```bash
    conda create --name <env_name> python=3.9
    conda activate <env_name>
    ```

3. **Install Dependencies**  

    With your conda environment active, install the required packages using the provided `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Code**

    The project is divided into different parts, each handling a specific data-loading pipeline. To run the models, execute the following commands:

    **For TF-IDF Reconstruction:**
    ```bash
    python3 naives_LR.py --method text
    ```

    **For Direct CSR Matrix Loading:**
    ```bash
    python3 naives_LR.py --method csr
    ```

    The `--method` flag selects the data-loading approach:
    - The **text** method reconstructs documents from the numeric data and applies TF-IDF transformation.
    - The **csr** method directly loads the data into CSR matrices.

    To run the CNN model, execute the following command:
    ```bash
    python3 cnn_classifier.py
    ```
    This script trains a Convolutional Neural Network on the 20 Newsgroups dataset using the reconstructed text data.

    Running these scripts will parse the dataset files, create sparse document-term matrices, train both the Naive Bayes and Logistic Regression models, and output detailed classification reports with metrics like precision, recall, F1-score, and support.

## Experimental Setup and Results

This section details two different data-loading pipelines tested on the 20 Newsgroups dataset. Both pipelines were used to train Naive Bayes and Logistic Regression classifiers on 20 newsgroups. Below, we outline the code, process, detailed output, and observations for each method along with key comparisons.

### Method 1: Text Reconstruction + TF-IDF

#### 1.1 Code & Process Explanation

**Data Parsing**  
- Read `vocabulary.txt` to obtain all unique words.  
- Load label files (`train.label` and `test.label`); each line contains a 1-based newsgroup ID that is converted to 0-based.  
- Determine the number of training and test documents.

**Reconstructing Each Document**  
- Call `load_numeric_data_as_dicts("train.data")` (and similarly for `test.data`) to parse data where each line (`doc_id word_id count`) is stored in a dictionary with `word_id` as key and `count` as value.  
- For each document, the function `reconstruct_document()` repeats the word from the vocabulary according to its count.  
  - *Example:* A document with `{3:2, 10:5}` might be reconstructed as:  
    `"myword3 myword3 myword10 myword10 myword10 myword10 myword10"` (note that the original order is lost).

**TF-IDF Vectorization**  
- Apply `TfidfVectorizer` (with potential stopword removal and lowercase conversion) to the reconstructed pseudo-text.  
- Transform the text into TF-IDF matrices (`X_train_tfidf` and `X_test_tfidf`).  
- Observed shapes: `(11269, 53666)` for training and `(7505, 53666)` for testing, nearly corresponding to the vocabulary size (minus filtered tokens).

**Training Naive Bayes & Logistic Regression**  
- Fit `MultinomialNB` on `X_train_tfidf` with `train_labels` and evaluate on `X_test_tfidf`.  
- Similarly, fit `LogisticRegression` (using `max_iter=1000`) and evaluate.  
- Generate classification reports showing precision, recall, F1-score, and support for each newsgroup.

#### 1.2 Detailed Output
**Naive Bayes:**  
- Achieved approximately 80% accuracy on 7505 test documents.  
- Some classes (e.g., `rec.sport.hockey` and `rec.sport.baseball`) reached precision/recall of 0.93–0.97, whereas classes like `talk.religion.misc` exhibited lower recall (around 0.14).

Naive Bayes Classification Report:
```
              precision    recall  f1-score   support

       alt.atheism       0.79      0.58      0.67       318
       comp.graphics       0.73      0.74      0.73       389
 comp.os.ms-windows.misc       0.77      0.70      0.73       391
comp.sys.ibm.pc.hardware       0.61      0.79      0.69       392
   comp.sys.mac.hardware       0.82      0.77      0.79       383
      comp.windows.x       0.86      0.76      0.80       390
      misc.forsale       0.89      0.75      0.81       382
         rec.autos       0.88      0.90      0.89       395
     rec.motorcycles       0.91      0.95      0.93       397
    rec.sport.baseball       0.94      0.93      0.93       397
    rec.sport.hockey       0.91      0.97      0.94       399
         sci.crypt       0.76      0.95      0.84       395
     sci.electronics       0.81      0.64      0.72       393
         sci.med       0.91      0.82      0.86       393
         sci.space       0.86      0.91      0.88       392
  soc.religion.christian       0.55      0.95      0.70       398
    talk.politics.guns       0.65      0.92      0.76       364
   talk.politics.mideast       0.92      0.91      0.92       376
    talk.politics.misc       0.93      0.49      0.64       310
    talk.religion.misc       0.92      0.14      0.25       251

        accuracy                           0.80      7505
         macro avg       0.82      0.78      0.77      7505
      weighted avg       0.82      0.80      0.79      7505
```

**Logistic Regression:**  
- Also achieved around 80% accuracy.  
- Categories such as `comp.graphics` achieved ~0.77 recall, while niche categories like `talk.politics.misc` scored lower metrics.


Logistic Regression Classification Report:
```
              precision    recall  f1-score   support

       alt.atheism       0.72      0.69      0.71       318
       comp.graphics       0.67      0.77      0.71       389
 comp.os.ms-windows.misc       0.75      0.70      0.72       391
comp.sys.ibm.pc.hardware       0.68      0.72      0.70       392
   comp.sys.mac.hardware       0.79      0.79      0.79       383
      comp.windows.x       0.83      0.74      0.78       390
      misc.forsale       0.72      0.83      0.77       382
         rec.autos       0.87      0.87      0.87       395
     rec.motorcycles       0.93      0.92      0.93       397
    rec.sport.baseball       0.88      0.91      0.89       397
    rec.sport.hockey       0.93      0.95      0.94       399
         sci.crypt       0.95      0.86      0.90       395
     sci.electronics       0.69      0.73      0.71       393
         sci.med       0.84      0.85      0.85       393
         sci.space       0.88      0.91      0.89       392
  soc.religion.christian       0.76      0.89      0.82       398
    talk.politics.guns       0.70      0.85      0.77       364
   talk.politics.mideast       0.95      0.84      0.89       376
    talk.politics.misc       0.75      0.57      0.64       310
    talk.religion.misc       0.74      0.41      0.53       251

        accuracy                           0.80      7505
         macro avg       0.80      0.79      0.79      7505
      weighted avg       0.80      0.80      0.80      7505
```

**Overall:** Both classifiers attain approximately 0.80 accuracy when using TF-IDF.

#### 1.3 Observations

- TF-IDF downweights very common words and emphasizes more discriminative terms.
- Although reconstructing text loses the original word order, the pseudo-text is effective for TF-IDF tokenization.
- The large vocabulary (~53k features) does not drastically harm performance but may increase memory requirements.

### Method 2: Direct CSR (Raw Counts)

#### 2.1 Code & Process Explanation

**Data Parsing**  
- Load `train.label` and `test.label` to obtain labels.  
- Read `train.data` and `test.data`, but instead of reconstructing text, directly store values in CSR matrices.

**CSR Matrix Construction**  
- For each line (`doc_id word_id count`), assign the count to index `(doc_id-1, word_id-1)` in a sparse matrix.
- Resulting shapes: `(11269, 61188)` for training and `(7505, 61188)` for testing.

**Raw Count Features**  
- Use the raw counts as feature values without converting them to TF-IDF.

**Model Training**  
- **Naive Bayes:** Train using `MultinomialNB().fit(X_train_csr, train_labels)` and predict on `X_test_csr`.  
- **Logistic Regression:** Train with `LogisticRegression(max_iter=1000).fit(X_train_csr, train_labels)` and evaluate similarly.

#### 2.2 Detailed Output

**Naive Bayes:**  
- Achieved an accuracy of around 0.78.  
- Some classes (e.g., `rec.motorcyc
les`, `rec.sport.hockey`) still performed well.

Naive Bayes Classification Report:
```
              precision    recall  f1-score   support

       alt.atheism       0.70      0.74      0.72       318
       comp.graphics       0.67      0.76      0.71       389
 comp.os.ms-windows.misc       0.82      0.53      0.64       391
comp.sys.ibm.pc.hardware       0.60      0.78      0.68       392
   comp.sys.mac.hardware       0.79      0.71      0.75       383
      comp.windows.x       0.81      0.78      0.80       390
      misc.forsale       0.91      0.59      0.72       382
         rec.autos       0.79      0.90      0.84       395
     rec.motorcycles       0.94      0.89      0.91       397
    rec.sport.baseball       0.96      0.87      0.91       397
    rec.sport.hockey       0.94      0.95      0.95       399
         sci.crypt       0.74      0.91      0.82       395
     sci.electronics       0.78      0.66      0.71       393
         sci.med       0.88      0.82      0.85       393
         sci.space       0.88      0.85      0.87       392
  soc.religion.christian       0.68      0.95      0.79       398
    talk.politics.guns       0.69      0.89      0.78       364
   talk.politics.mideast       0.88      0.86      0.87       376
    talk.politics.misc       0.57      0.59      0.58       310
    talk.religion.misc       0.83      0.35      0.50       251

        accuracy                           0.78      7505
         macro avg       0.79      0.77      0.77      7505
      weighted avg       0.80      0.78      0.78      7505
```

Logistic Regression Classification Report:
```
              precision    recall  f1-score   support

       alt.atheism       0.65      0.67      0.66       318
       comp.graphics       0.59      0.69      0.63       389
 comp.os.ms-windows.misc       0.70      0.62      0.66       391
comp.sys.ibm.pc.hardware       0.63      0.67      0.65       392
   comp.sys.mac.hardware       0.70      0.73      0.72       383
      comp.windows.x       0.74      0.65      0.69       390
      misc.forsale       0.73      0.84      0.78       382
         rec.autos       0.74      0.77      0.76       395
     rec.motorcycles       0.88      0.87      0.87       397
    rec.sport.baseball       0.79      0.82      0.80       397
    rec.sport.hockey       0.89      0.89      0.89       399
         sci.crypt       0.83      0.78      0.81       395
     sci.electronics       0.60      0.63      0.62       393
         sci.med       0.78      0.71      0.74       393
         sci.space       0.86      0.81      0.84       392
  soc.religion.christian       0.76      0.81      0.78       398
    talk.politics.guns       0.63      0.76      0.69       364
   talk.politics.mideast       0.88      0.72      0.79       376
    talk.politics.misc       0.63      0.48      0.54       310
    talk.religion.misc       0.50      0.48      0.49       251

        accuracy                           0.73      7505
         macro avg       0.73      0.72      0.72      7505
      weighted avg       0.73      0.73      0.73      7505
```

#### 2.3 Observations

- Direct raw counts provide a simpler and memory-efficient pipeline.
- The CSR approach underperforms compared to the TF-IDF method, particularly for Logistic Regression which benefits from feature weighting.

#### Key Differences & Overall Observations

**Accuracy:**  
- TF-IDF Pipeline: Approximately 80% accuracy for both Naive Bayes and Logistic Regression.  
- CSR (Raw Counts) Pipeline: Approximately 78% (Naive Bayes) and 73% (Logistic Regression).

**Reasons for TF-IDF Benefit:**  
- TF-IDF naturally downweights common words and upweights discriminative ones, which particularly improves Logistic Regression performance.

**Ease of Implementation:**  
- The CSR method is straightforward and more memory-efficient, making it preferable for very large datasets.  
- Text reconstruction with TF-IDF is a more traditional approach that offers higher accuracy, albeit with increased computational cost.

**Conclusion**:

Both Naive Bayes and Logistic Regression perform well on the 20 Newsgroups dataset. The text reconstruction combined with TF-IDF yields about 80% accuracy for both models, demonstrating the effectiveness of TF-IDF weighting in improving text classification. In contrast, the direct CSR approach using raw counts is simpler and more efficient in memory usage but provides lower accuracy, especially for Logistic Regression.

### Method 3. Problem & Approach for CNN

We want to classify **20 Newsgroups** documents—available only as `(doc_id, word_id, count)` files—into one of **20** categories. Since a standard CNN for text typically needs **token sequences**, we:

1. **Reconstruct** approximate text from `(doc_id, word_id, count)` by repeating each vocabulary word by its count.
2. **Preprocess** the resulting text (convert to lowercase, remove punctuation/numbers, filter out stopwords).
3. **Tokenize** into integer sequences and **pad** each document to a fixed length.
4. **Train a CNN** with an **embedding** layer, **Conv1D**, and **GlobalMaxPooling1D** to learn local n-gram features.
5. **Evaluate** on the test set to produce classification metrics (accuracy, precision, recall, F1-score).

**Why this method**:
- A CNN with an embedding layer captures local word patterns effectively.
- Reconstructing text enables typical text-based preprocessing rather than relying solely on bag-of-words counts.

### Code Summary

#### 3.1 Data Preparation

1. **Load Vocabulary & Labels**:  
  - Read `vocabulary.txt`, `train.map` (for mapping label IDs to newsgroup names), and the `train`/`test.label` files.

2. **Reconstruct Text**:  
  - For each document, gather `(word_id, count)` pairs and form a “pseudo-document” by repeating `vocabulary[word_id]` `count` times.

3. **Preprocessing**:  
  - Convert the text to lowercase, remove punctuation, split on whitespace, and filter out stopwords.

4. **Tokenize & Pad**:  
  - Utilize a Keras `Tokenizer` to convert tokens into integer sequences, then apply `pad_sequences` to ensure a uniform sequence length (e.g., 400 tokens per document).

#### 3.2 CNN Model

1. **Embedding Layer**:  
  - Transforms token IDs into 100-dimensional vectors.

2. **Conv1D Layer**:  
  - Applies 128 filters with a kernel size of 5 to detect local n-gram patterns.

3. **GlobalMaxPooling1D**:  
  - Aggregates the most salient feature for each filter across the sequence.

4. **Dense Layers with Dropout**:  
  - Used for the final classification into 20 newsgroups.

5. **Training**:  
  - The model is trained using the `Adam` optimizer and `sparse_categorical_crossentropy` loss for 5 epochs with a batch size of 64.

#### 3.3 Evaluation

- The model is evaluated on the test set (7,505 documents) by reporting overall test accuracy and a detailed classification report (precision, recall, and F1-score per class).

#### 3.4 Key Results & Observations

#### Training/Validation Performance
- **Rapid Learning**: The model quickly learns to classify documents, achieving high training accuracy.
- **Validation Plateau**: Although validation accuracy remains low during training, the final test accuracy stabilizes around **71–72%**.

#### Output Logs

```bash
week 5/cnn_classifier.py
[nltk_data] Downloading package stopwords...
[nltk_data]   Unzipping corpora/stopwords.zip.
[INFO] Loaded vocabulary. Size = 61188
[INFO] Newsgroups: ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', ... 'talk.religion.misc']
[INFO] Number of training documents = 11269
[INFO] Number of test documents = 7505
[DEBUG] First training doc (partial): archive archive archive archive name name atheism atheism ...
[DEBUG] Sample tokens from train doc 0: ['archive', 'archive', 'archive', 'archive', 'name', 'name', ...]
[INFO] Train sequence shape = (11269, 400)
[INFO] Test sequence shape  = (7505, 400)
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 400, 100)          5383200
 conv1d (Conv1D)             (None, 396, 128)          64128
 global_max_pooling1d (GlobalMaxPooling1D) (None, 128)   0
 dropout (Dropout)           (None, 128)               0
 dense (Dense)               (None, 64)                8256
 dense_1 (Dense)             (None, 20)                1300
=================================================================
Total params: 5,456,884
Trainable params: 5,456,884
Non-trainable params: 0
_________________________________________________________________
Epoch 1/5
159/159 [==============================] - 9s 53ms/step - loss: 2.7908 - accuracy: 0.1501 - val_loss: 6.7499 - val_accuracy: 0.0000e+00
Epoch 2/5
159/159 [==============================] - 9s 54ms/step - loss: 1.4529 - accuracy: 0.6122 - val_loss: 10.7681 - val_accuracy: 0.1775
Epoch 3/5
159/159 [==============================] - 9s 54ms/step - loss: 0.6359 - accuracy: 0.8340 - val_loss: 14.0124 - val_accuracy: 0.1846
Epoch 4/5
159/159 [==============================] - 9s 56ms/step - loss: 0.3130 - accuracy: 0.9254 - val_loss: 16.1429 - val_accuracy: 0.1979
Epoch 5/5
159/159 [==============================] - 8s 53ms/step - loss: 0.1831 - accuracy: 0.9562 - val_loss: 18.1414 - val_accuracy: 0.1988
[RESULT] CNN Test Accuracy: 0.7190
```

#### Classification Report

```bash
CNN Classification Report:
              precision    recall  f1-score   support

       alt.atheism       0.63      0.69      0.66       318
       comp.graphics       0.64      0.70      0.67       389
 comp.os.ms-windows.misc       0.70      0.66      0.68       391
comp.sys.ibm.pc.hardware       0.59      0.74      0.66       392
   comp.sys.mac.hardware       0.80      0.78      0.79       383
      comp.windows.x       0.79      0.69      0.74       390
      misc.forsale       0.71      0.72      0.72       382
         rec.autos       0.73      0.79      0.76       395
     rec.motorcycles       0.88      0.91      0.89       397
    rec.sport.baseball       0.88      0.90      0.89       397
    rec.sport.hockey       0.93      0.93      0.93       399
         sci.crypt       0.83      0.86      0.84       395
     sci.electronics       0.53      0.61      0.57       393
         sci.med       0.71      0.72      0.71       393
         sci.space       0.83      0.82      0.82       392
  soc.religion.christian       0.59      0.87      0.70       398
    talk.politics.guns       0.55      0.86      0.67       364
   talk.politics.mideast       0.86      0.70      0.77       376
    talk.politics.misc       0.00      0.00      0.00       310
    talk.religion.misc       0.00      0.00      0.00       251

        accuracy                           0.72      7505
         macro avg       0.66      0.70      0.67      7505
      weighted avg       0.68      0.72      0.69      7505
```

> **Notes**:
> - Categories such as `talk.politics.misc` and `talk.religion.misc` have **0 predicted samples**, resulting in 0.0 F1-scores.
> - Sports categories (e.g., `rec.sport.hockey`, `rec.motorcycles`) show high precision and recall.
> - Overall test accuracy is approximately **72%**.


#### 3.5 Conclusion & Future Directions

**Conclusion**:
- A sequence-based CNN using an embedding layer and Conv1D over tokenized documents achieves approximately **72% test accuracy** on the 20 Newsgroups dataset, even with only approximate text reconstruction.
- Most newsgroups are classified accurately, though some classes are not well-predicted.

**Potential Improvements**:
- Enhance text reconstruction or use the original text to better preserve word order.
- Apply class weighting to improve predictions for underrepresented categories.
- Perform further hyperparameter tuning (e.g., try different kernel sizes, extend training epochs, incorporate pre-trained embeddings).
- Limit vocabulary size or employ subword tokenization to reduce overfitting.

Overall, this CNN approach demonstrates that sequence-based learning can effectively capture local textual patterns and can outperform traditional bag-of-words models even with partially reconstructed text.

## Summary

- **Data Files:**  
  The dataset uses a numeric representation of text documents. The vocabulary, mapping file (**train.map**), label files, and document-word count files allow us to reconstruct a document-term matrix suitable for machine learning.

- **Models Implemented:**  
  - **Naive Bayes:** A generative model that uses word counts and strong independence assumptions.  
  - **Logistic Regression:** A discriminative model that directly models the probability of a label given the input features.  
  - **Convolutional Neural Network:** A deep learning model that learns local n-gram features from tokenized text sequences.

- **Evaluation:**  
  Both models are evaluated using a classification report detailing performance metrics for each newsgroup class.

### Comparison & Key Insights

- **Overall Accuracy:**  
  - Naive Bayes & Logistic Regression both peak around 80% with TF-IDF.  
  - CNN sits around 72% with partially reconstructed text. True raw text might yield higher CNN performance.

- **Implementation Complexity:**  
  - NB and LR (with CSR or TF-IDF) are straightforward, requiring less data preprocessing.  
  - CNN needs a full text reconstruction → tokenization → embedding pipeline, which is more complex.

- **Memory & Computation:**  
  - TF-IDF or raw count matrices can be large but remain feasible in a sparse format for NB/LR.  
  - CNN involves large embedding matrices (millions of parameters) and heavier GPU/CPU usage during training.

- **Best-Performing Model:**  
  - TF-IDF combined with Logistic Regression or Naive Bayes emerges as the top performer on this dataset.  
  - The CNN’s sequence approach shows promise (especially on distinct categories) but is currently outperformed by well-tuned classical models.

- **Future Directions:**  
  - If original raw text is available, a CNN or other deep learning model might outperform classical methods by capturing richer context.  
  - Additional hyperparameter tuning, class weighting, or pre-trained embeddings could help CNN results approach or surpass 80%.