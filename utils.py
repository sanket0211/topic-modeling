from codecs import unicode_escape_decode
from lib2to3.pytree import LeafPattern
from re import U
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import operator
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import spacy
from scipy import spatial
import ast

STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS=['!','(',')','-','[',']','{','}',';',':',"'",'"','\\',',','<','>','.','/','?','@','#','$','%','^','&','*','_','~', '...', "’", "‘"]
NOISE=["redacted"]
LEMMATIZER = WordNetLemmatizer()
class Utils:
    """
    Model Utility Methods
    """
        
    def sorting_list_tuples(self, sub_li):
        sub_li.sort(key = lambda x: x[1])
        sub_li.reverse()
        return (sub_li)
               
    
    def filter_pos(self, text):
        tokenized_text = word_tokenize(text)
        tagged = nltk.pos_tag(tokenized_text)
        final = []
        for i,j in tagged:
            if (("NN" in j) or ("RB" in j) or ("JJ" in j) or ("VB" in j)):
                final.append(i)
        final = " ".join(final)
        return final
    
    def lemmatization(self, text):
        tokenized_text = word_tokenize(text)
        final=[]
        for token in tokenized_text:
            final.append(LEMMATIZER.lemmatize(token))
        final = " ".join(final)
        return final
        
    
    def get_trigram_frequency(self, documents):
        processed_documents = self.preprocess_docs(documents)
        trigram_freq_dict = {}
        for doc in processed_documents:
            tokenized_doc = word_tokenize(doc)
            trigrams = list(ngrams(tokenized_doc,3))
            for trigram in trigrams:
                if trigram not in trigram_freq_dict:
                    trigram_freq_dict[trigram] = 0
                trigram_freq_dict[trigram]+=1
        sorted_trigram_freq_dict = dict( sorted(trigram_freq_dict.items(), key=operator.itemgetter(1),reverse=True))
        cnt=0
        top_100=[]
        for key in sorted_trigram_freq_dict:
            cnt+=1
            top_100.append(key)
            if cnt>50:
                break 
        return top_100
    
    def get_bigram_frequency(self, documents):
        processed_documents = self.preprocess_docs(documents)
        bigram_freq_dict = {}
        for doc in processed_documents:
            tokenized_doc = word_tokenize(doc)
            bigrams = list(ngrams(tokenized_doc,2))
            for bigram in bigrams:
                if bigram not in bigram_freq_dict:
                    bigram_freq_dict[bigram] = 0
                bigram_freq_dict[bigram]+=1
        sorted_bigram_freq_dict = dict( sorted(bigram_freq_dict.items(), key=operator.itemgetter(1),reverse=True))
        cnt=0
        top_100=[]
        for key in sorted_bigram_freq_dict:
            cnt+=1
            top_100.append(key)
            if cnt>50:
                break 
        return top_100
    
    def get_top_100_freq_words(self, documents):
        processed_documents = self.preprocess_docs(documents)
        word_freq_dict = {}
        for doc in processed_documents:
            tokenized_doc = word_tokenize(doc)
            for token in tokenized_doc:
                if token not in word_freq_dict:
                    word_freq_dict[token]=0
                word_freq_dict[token]+=1
        sorted_word_freq_dict = dict( sorted(word_freq_dict.items(), key=operator.itemgetter(1),reverse=True))
        cnt=0
        top_100=[]
        for key in sorted_word_freq_dict:
            cnt+=1
            top_100.append(key)
            if cnt>100:
                break
        return top_100
    
    def prepare_multilabel_classification_data(self, df):
        doc_tag_dict={}
        docs = df['doc']
        topics = df['label']
        for doc, top in zip(docs, topics):
            if doc not in doc_tag_dict:
                doc_tag_dict[doc]=[]
            if top not in doc_tag_dict[doc]:
                doc_tag_dict[doc].append(top)
        unique_topics = list(set(topics))
        doc_label_binary_dict={}
        for doc in doc_tag_dict:
            if doc not in doc_label_binary_dict:
                doc_label_binary_dict[doc]=[]
            for top in unique_topics:
                if top in doc_tag_dict[doc]:
                    doc_label_binary_dict[doc].append(1)
                else:
                    doc_label_binary_dict[doc].append(0)
        return doc_label_binary_dict, unique_topics




    def plot_histogram(self, documents):
        document_len_frequency = {}
        for doc in documents:
            tokenized_doc = word_tokenize(doc)
            if len(tokenized_doc) not in document_len_frequency:
                document_len_frequency[len(tokenized_doc)]=0
            document_len_frequency[len(tokenized_doc)]+=1
        return document_len_frequency
    
    #gives us total no. of documents, average length of each document, no. of unique tokens
    def data_stats(self, documents):
        no_of_documents = len(documents)
        avg_len_of_document = 0
        unique_tokens = []
        for doc in documents:
            tokenized_doc = word_tokenize(doc)
            avg_len_of_document += len(tokenized_doc)
            unique_tokens.extend(tokenized_doc)
        unique_tokens = list(set(unique_tokens))
        avg_len_of_document /= no_of_documents
        return no_of_documents, avg_len_of_document, len(unique_tokens)
    
    def load_data(self, file):
        df = pd.read_csv(file, header=None, names=['documents', 'colB'], index_col=False)
        df.drop("colB", axis=1, inplace=True)
        return df

    def remove_stopwords(self,text):
        tokenized_text = word_tokenize(text)
        final=[]
        for w in tokenized_text:
            if (w not in STOPWORDS) and (w not in NOISE):
                final.append(w)
        final = " ".join(final)
        return final
        
    def tokenize(self, documents):
        final = []
        for doc in documents:
            tokenized_text = word_tokenize(doc)
            final.append(tokenized_text)
        return final
        
    def remove_punctuations(self,text):
        final = [w for w in text if not w.lower() in PUNCTUATIONS]
        
        final = "".join(final)
        return final

    def get_unique_topics_freq(self, df):
        label_freq={}
        labels = df['label']
        for i in labels:
            if i not in label_freq:
                label_freq[i]=0
            label_freq[i]+=1
        return label_freq

    def preprocess_docs(self, documents):
        final=[]
        for doc in documents:
            doc = doc.lower()
            doc = doc.replace("n't", " not")
            doc = self.filter_pos(doc)
            doc = doc.replace(",", " ")
            doc = self.remove_punctuations(doc)
            doc = self.remove_stopwords(doc)
            doc = self.lemmatization(doc)
            #doc = doc.replace("  ", " ")
            final.append(doc)
        return final    