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

    The code in this project is split into parts. For the Naive Bayes and Logistic Regression models, run the script:
    ```bash
    python naives_LR.py
    ```

    This script will parse the data files, build sparse document-term matrices, train the models, and output classification reports that include metrics such as precision, recall, f1-score, and support for each newsgroup class.

## Summary

- **Data Files:**  
  The dataset uses a numeric representation of text documents. The vocabulary, mapping file (**train.map**), label files, and document-word count files allow us to reconstruct a document-term matrix suitable for machine learning.

- **Models Implemented:**  
  - **Naive Bayes:** A generative model that uses word counts and strong independence assumptions.  
  - **Logistic Regression:** A discriminative model that directly models the probability of a label given the input features.

- **Evaluation:**  
  Both models are evaluated using a classification report detailing performance metrics for each newsgroup class.




