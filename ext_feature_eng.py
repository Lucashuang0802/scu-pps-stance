from feature_engineering import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

def tf_idf_features(headlines, bodies):
    X = []
    for _, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_bodies = []
        sentences = sent_tokenize(body)

        for sentence in sentences:
            clean_body = clean(sentence)
            clean_bodies.append(" ".join(get_tokenized_lemmas(clean_body)))

        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_headline = " ".join(clean_headline)   

        tfidf_vectorizer = TfidfVectorizer(min_df = 1)
        tfidf_matrix = tfidf_vectorizer.fit_transform(clean_bodies + [clean_headline])
        cosine = cosine_similarity(tfidf_matrix[len(clean_bodies)], tfidf_matrix) 
        the_cosine = cosine[0]
        winner = max(the_cosine[:-1])
        X.append([winner])

    return X
