import sys
import nltk
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
import json
import ldaconfig
sys.path.append('../../') 
from utils import Utils
utils_obj = Utils()

def main(args)
    df = utils_obj.load_data(args.file_path)
    documents=df['documents'].tolist()
    processed_documents = utils_obj.preprocess_docs(documents)
    processed_documents_words = utils_obj.tokenize(processed_documents)

    bigrams_phrases = gensim.models.Phrases(processed_documents_words, min_count=5, threshold=50)
    trigrams_phrases = gensim.models.Phrases(bigrams_phrases[processed_documents_words], min_count=5, threshold=50)

    bigram = gensim.models.phrases.Phraser(bigrams_phrases)
    trigram = gensim.models.phrases.Phraser(trigrams_phrases)

    def create_bigrams(texts):
        return ([bigram[doc] for doc in texts])

    def create_trigrams(texts):
        return ([trigram[bigram[doc]] for doc in texts])

    data_bigrams = create_bigrams(processed_documents_words)
    data_bigrams_trigrams = create_trigrams(data_bigrams)

    id2word = corpora.Dictionary(data_bigrams_trigrams)
    texts = data_bigrams_trigrams
    corpus = [id2word.doc2bow(text) for text in texts]

    tfidf = TfidfModel(corpus, id2word=id2word)

    low_value = 0.03
    words=[]
    words_missing_in_tfidf = []

    for i in range(0, len(corpus)):
        bow = corpus[i]
        low_value_words = []
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words+words_missing_in_tfidf
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]
        
        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        corpus[i] = new_bow

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=ldaconfig.NUM_TOPICS,
                                            random_state=1,
                                            chunksize=100,
                                            passes=20,
                                            alpha="auto")

    topics = lda_model.show_topics(num_words=25, formatted=False, num_topics=15)

    topics_dict={}
    for topic in topics:
        topics_dict[str(topic[0])]=str(topic[1])

    with open(args.outputfile_path, "w") as outfile:
        json.dump(topics_dict, outfile)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_path', type=str, required=True)
  parser.add_argument('--outputfile_path', type=str, required=True)
  args = parser.parse_args()
  main(args)