## topic-modeling

1. Here we explore TfIdf, LDA methods for getting topic clusters.

## Installation

```bash
pip install -r requirements.txt
!python3 -m spacy download en_core_web_sm
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```
You can then run scripts as follows
1. Dataset folder contains scripts to visualize and analyze the Dataset.
2. Subtask1/Tfidf contains scripts to extract clusters using the Tfidf method. 
3. Subtask1/LDA contains scripts to extract clusters using the LDA method. 