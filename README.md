# Machine Learning Assignment: 20 Newsgroups Classification

## Introduction

""

## How to Download the Data

The data can be downloaded from the following URL:  
[20 Newsgroups Dataset](http://people.csail.mit.edu/jrennie/20Newsgroups/)

## Data Description

The dataset is organized into six files:

1. **vocabulary.txt**  
    A list of words that may appear in documents. The line number corresponds to the word's ID. For example, the first word (*archive*) has wordId 1, the second (*name*) has wordId 2, and so on.

2. **newsgrouplabels.txt**  
    A list of newsgroups representing the origin of each document. The line number corresponds to the label's ID used in the label files. For instance, the first line (*alt.atheism*) has ID 1.

3. **train.label**  
    Each line contains the label for one document from the training set. The document ID (docId) corresponds to the line number.

4. **test.label**  
    Similar to `train.label`, this file contains the labels for the test documents.

5. **train.data**  
    This file specifies the counts for each word in each training document. Each line is formatted as:  

    ```
    docId wordId count
    ```

    This indicates that the training document with ID `docId` contains the word with ID `wordId` repeated `count` times. Any word/document combination not included in this file is assumed to have a count of 0.

6. **test.data**  
    This file follows the same format as `train.data` but contains word counts for the test documents.

## Getting Started

1. **Clone the Repository**  
    Clone the repository and navigate to the project directory:
    ```bash
    git clone <repository-url>
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