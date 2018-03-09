from feature_engineering import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import vstack
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

import statistics as s

import pdb

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

        all_text = " ".join(clean_bodies + [clean_headline])
        vec = TfidfVectorizer(ngram_range=(1, 3))
        vec.fit([all_text])

        xHeadlineTfidf = vec.transform([clean_headline])
        clean_bodies = " ".join(clean_bodies)
        xBodyTfidf = vec.transform([clean_bodies])

        cosine = cosine_similarity(xHeadlineTfidf, xBodyTfidf)[-1][-1]
        X.append([cosine])

    return X

def svd_features(headlines, bodies):
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

        all_text = " ".join(clean_bodies + [clean_headline])
        vec = TfidfVectorizer(ngram_range=(1, 3))
        vec.fit([all_text])

        xHeadlineTfidf = vec.transform([clean_headline])
        clean_bodies = " ".join(clean_bodies)
        xBodyTfidf = vec.transform([clean_bodies])

        svd = TruncatedSVD(n_components=1, n_iter=15)
        xHBTfidf = vstack([xHeadlineTfidf, xBodyTfidf])
        svd.fit(xHBTfidf)
        xHeadlineSvd = svd.transform(xHeadlineTfidf)
        xBodySvd = svd.transform(xBodyTfidf)

        cosine = cosine_similarity(xHeadlineSvd, xBodySvd)[-1][-1]
        X.append([cosine])

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
        X.append([b_neg, b_neu, b_pos, b_compound, h_neg, h_neu, h_pos, h_compound])

    return X
