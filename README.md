# Machine Learning Project: Newsgroups Classification using Generative and Discriminative Models

## Introduction

*This project focuses on classifying newsgroup posts using two types of models: a generative model (Naive Bayes) and a discriminative model (Logistic Regression). We use the 20 Newsgroups dataset to train, evaluate, and compare the performance of these models.*

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

### Key Differences & Overall Observations

**Accuracy:**  
- TF-IDF Pipeline: Approximately 80% accuracy for both Naive Bayes and Logistic Regression.  
- CSR (Raw Counts) Pipeline: Approximately 78% (Naive Bayes) and 73% (Logistic Regression).

**Reasons for TF-IDF Benefit:**  
- TF-IDF naturally downweights common words and upweights discriminative ones, which particularly improves Logistic Regression performance.

**Ease of Implementation:**  
- The CSR method is straightforward and more memory-efficient, making it preferable for very large datasets.  
- Text reconstruction with TF-IDF is a more traditional approach that offers higher accuracy, albeit with increased computational cost.

### Conclusion

Both Naive Bayes and Logistic Regression perform well on the 20 Newsgroups dataset. The text reconstruction combined with TF-IDF yields about 80% accuracy for both models, demonstrating the effectiveness of TF-IDF weighting in improving text classification. In contrast, the direct CSR approach using raw counts is simpler and more efficient in memory usage but provides lower accuracy, especially for Logistic Regression.

## Summary

- **Data Files:**  
  The dataset uses a numeric representation of text documents. The vocabulary, mapping file (**train.map**), label files, and document-word count files allow us to reconstruct a document-term matrix suitable for machine learning.

- **Models Implemented:**  
  - **Naive Bayes:** A generative model that uses word counts and strong independence assumptions.  
  - **Logistic Regression:** A discriminative model that directly models the probability of a label given the input features.

- **Evaluation:**  
  Both models are evaluated using a classification report detailing performance metrics for each newsgroup class.




