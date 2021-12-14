from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfVectorizer:

    def tfidf_vectorizer(corpus, ngram_range=(1, 2), max_features=10000):
        """ return vectors of each document in corpus """
        vectorizer = TfidfVectorizer(
            min_df=5, ngram_range=ngram_range, max_features=max_features)
        X = vectorizer.fit_transform(corpus)
        return X, vectorizer

    def averaging():
        pass

    def vec_from_channel(self, channel_id):
        pass

    def channel_from_vec(self, emb):
        """ most similar """
        pass
