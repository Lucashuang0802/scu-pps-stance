from feature_engineering import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

import statistics as s

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

def sentiment_features(headlines, bodies):
    X = []
    sid = SentimentIntensityAnalyzer()

    for _, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        
        def compute_sentiment(sentences):
            result = []
            for sentence in sentences:
                vs = sid.polarity_scores(sentence)
                result.append(vs)
            return pd.DataFrame(result).mean()

        body_sent_score = compute_sentiment(sent_tokenize(body))
        b_neg = body_sent_score['neg']
        b_neu = body_sent_score['neu']
        b_pos = body_sent_score['pos']
        b_compound = body_sent_score['compound']
        headline_sent_score = sid.polarity_scores(headline)
        h_neg = headline_sent_score['neg']
        h_neu = headline_sent_score['neu']
        h_pos = headline_sent_score['pos']
        h_compound = headline_sent_score['compound']
        X.append([b_neg, b_neu, b_pos, b_compound, h_neg, h_neg, h_pos, h_compound])

    return X
