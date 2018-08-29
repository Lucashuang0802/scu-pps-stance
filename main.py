import numpy as np

# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from feature_engineering import hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from ext_feature_eng import tf_idf_features, sentiment_features

from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds, get_related_stances_for_folds
from utils.score import report_score, LABELS, score_submission, detailed_score
from utils.system import parse_params, check_version, check_results_dir
import sys

np.set_printoptions(threshold=np.inf)

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X_tf_idf = gen_or_load_feats(tf_idf_features, h, b, "features/tf_idf."+name+".npy")
    X_sentiment = gen_or_load_feats(sentiment_features, h, b, "features/sentiment."+name+".npy")

    X = np.c_[X_hand, X_overlap, X_tf_idf, X_sentiment]
    return X, y

if __name__ == "__main__":
    check_version()
    a = parse_params()
    check_results_dir()

    # Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)

    if a:
        fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)
    else:
        fold_stances, hold_out_stances = get_related_stances_for_folds(d, folds, hold_out)

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))

    best_score = 0
    best_fold = None

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        # clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf = RandomForestClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

    # Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted, type='dev')
    print("Break down scores")
    detailed_score(actual, predicted)

    print("")
    print("")

    # Load and run on competition dataset
    competition_dataset = DataSet("competition_test")
    if a:
        X_competition, y_competition = generate_features(competition_dataset.related_stance, competition_dataset,
                                                         "competition")
    else:
        X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset,
                                                         "competition")

    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted, type='test')

    # save the model to disk
    # import pickle
    # if best_fold:
    #     filename = 'finalized_model.sav'
    #     pickle.dump(best_fold, open(filename, 'wb'))

    print("Break down scores")
    detailed_score(actual,predicted)
