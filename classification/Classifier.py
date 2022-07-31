import pandas as pd
import numpy as np
import argparse
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score,hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from skmultilearn.problem_transform import BinaryRelevance
import sys
sys.path.append("../subtask1/TfIdf/")
import tfidfconfig
sys.path.append("../")
from utils import Utils
util_obj = Utils()

def main(args):
    df = pd.read_csv(args.file_path, sep='\t')
    corpus = util_obj.preprocess_docs(df['text'])
    vectorizer = TfidfVectorizer(lowercase=True,
                                    max_features=tfidfconfig.MAX_FEATURES,
                                    max_df=tfidfconfig.MAX_DF,
                                    min_df=tfidfconfig.MIN_DF,
                                    ngram_range=(1,3),
                                    stop_words="english"
                                )
    Xfeatures = vectorizer.fit_transform(corpus).toarray()
    y = df[['delivery punctuality', 'mobile fitter', 'value for money', 'wait time', 'garage service', 'ease of booking', 'booking confusion', 'None', 'location']]
    X_train,X_test,y_train,y_test = train_test_split(Xfeatures,y,test_size=0.3,random_state=42)
    binary_rel_clf = BinaryRelevance(MultinomialNB())
    binary_rel_clf.fit(X_train,y_train)
    br_prediction = binary_rel_clf.predict(X_test)
    br_prediction.toarray()
    print(accuracy_score(y_test,br_prediction))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_path', type=str, required=True)
  #parser.add_argument('--outputfile_path', type=str, required=True)
  args = parser.parse_args()
  main(args)