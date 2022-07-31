import sys
import argparse
import nltk
import numpy as np
import matplotlib.pyplot as plt
import operator
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tfidfconfig
sys.path.append('../../') 
from utils import Utils
utils_obj = Utils()

def main(args):
    df = utils_obj.load_data(args.file_path)
    documents = df['documents']
    processed_documents = utils_obj.preprocess_docs(documents)

    vectorizer = TfidfVectorizer(lowercase=True,
                                    max_features=tfidfconfig.MAX_FEATURES,
                                    max_df=tfidfconfig.MAX_DF,
                                    min_df=tfidfconfig.MIN_DF,
                                    ngram_range=(1,3),
                                    stop_words="english"
                                )
    vectors = vectorizer.fit_transform(processed_documents)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    all_keywords = []
    for description in denselist:
        x=0
        keywords=[]
        for word in description:
            if word>0:
                keywords.append(feature_names[x])
            x+=1
        all_keywords.append(keywords)

    true_k = tfidfconfig.NO_OF_CLUSTERS
    model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)
    model.fit(vectors)
    order_centroids = model.cluster_centers_.argsort()[:,::-1]
    terms = vectorizer.get_feature_names()

    with open(args.outputfile_path, "w", encoding="utf-8") as f:
        for i in range(true_k):
            f.write(f"Cluster {i}")
            f.write("\n")
            for ind in order_centroids[i,:15]:
                f.write (' %s' % terms[ind],)
                f.write("\n")
            f.write("\n")
            f.write("\n")

    kmean_indices = model.fit_predict(vectors)
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(vectors.toarray())
    colours = ["r", "b", "c", "y", "m"]
    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]

    fig, ax = plt.subplots(figsize=(50,50))
    #ax.scatter(x_axis, y_axis, c=kmean_indices, cmap='Set2')
    ax.scatter(x_axis, y_axis, c=[colours[d] for d in kmean_indices])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_path', type=str, required=True)
  parser.add_argument('--outputfile_path', type=str, required=True)
  args = parser.parse_args()
  main(args)