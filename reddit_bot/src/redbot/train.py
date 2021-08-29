from datetime import datetime
import time
from urllib.parse import urlparse

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn import compose, feature_extraction, linear_model, metrics, model_selection
from xgboost import XGBClassifier

from redbot import db


def preprocess_valid_data(con):
    """Replace urls with domain names and highrank24 with 'hot or not' indicator.

    Generate a raw dataframe from all valid posts (created over 24 hours ago).
    Preprocess the raw dataframe to generate a processed dataframe with
    features (domain name, title, score, up_vote ratio) and a 'hot or not' indicator
    for each valid post.

    Args:
        con: database connection.

    Returns:
        df: DataFrame. A processed dataframe with features (domain name, title, score, up_vote ratio)
        and a 'hot or not' indicator for each valid post.
    """
    current_time = time.time()
    current_time_utc = datetime.utcfromtimestamp(current_time)

    raw_df = db.retrieve_valid_posts_for_training(con, current_time_utc)

    df = raw_df.copy()
    full_domain = []
    hot_or_not = []
    for idx in df.index:
        domain = urlparse(df.loc[idx]['url'])[1]
        if domain.split('.')[0] == 'www':
            full_domain.append(domain.split('.')[1])
        else:
            full_domain.append(domain.split('.')[0])

        if np.isnan(df.loc[idx]['highrank24']):
            hot_or_not.append(0)
        else:
            hot_or_not.append(1)

    df['domain'] = full_domain
    df['hot_or_not'] = hot_or_not

    df.drop(['url', 'highrank24'], axis=1, inplace=True)
    return df


def create_train_test_split(df, random_sample=False):
    """Create training and test features and labels for model training and testing.

    Either create a randomized train-test split or a sequence-aware split
    where first 80% valid posts are used for training and the latter 20% for testing.

    Args:
         df: DataFrame. Dataframe of features and 'hot or not' indicators for all valid posts.
         random_sample: bool. If True, use randomized split; if False use sequence-aware split.

    Returns:
        x_train: DataFrame. DataFrame of features for training model.
        y_train: Series. Series of labels for training model.
        x_test: DataFrame. DataFrame of features for testing/evaluating model.
        y_test: Series. Series of true labels for testing/evaluating model.
    """
    if random_sample:
        train_idx, test_idx = model_selection.train_test_split(df.index, test_size=0.2, stratify=df['hot_or_not'])
    else:
        total_idx = len(df.index)
        boundary = total_idx * 8 // 10
        train_idx = list(range(boundary))
        test_idx = list(range(boundary, total_idx))

    train_df = df.loc[train_idx][:]
    test_df = df.loc[test_idx][:]

    y_train = train_df['hot_or_not']
    x_train = train_df.drop(['hot_or_not'], axis=1)

    y_test = test_df['hot_or_not']
    x_test = test_df.drop(['hot_or_not'], axis=1)

    return x_train, y_train, x_test, y_test


def transform_text_columns(x_train, x_test):
    """Transform textual features into numerical representation.

    Use Tfidf vectorizer to transform titles and domains in train and test features
    into numerical representation.  The remainder of the numerical features (scores, up_vote_ratios)
    are used without any transformation.

    Args:
        x_train: DataFrame. DataFrame of features (combination of text features: titles and domains,
        and numerical features: scores and up_vote ratios), for training posts.
        x_test: DataFrame.  DataFrame of the same categories of features, for test posts.

    Returns:
        x_train_transformed: sparse matrix. For training posts, text features converted into sparse numerical matrix;
        numerical features preserved unmodified.
        x_test_transformed: sparse matrix. Same transformation applied to text features of test posts; numerical features
        preserved unmodified.
    """
    ct = compose.ColumnTransformer(
        [('titles_bow',
          feature_extraction.text.TfidfVectorizer(stop_words=['of', 'to', 'in', 'for', 'the', 'and', 'are', 'on',
                                                              'as', 'from', 'is', 'he', 'his', 'it', 'that', 'was',
                                                              'with']), 'title'),
         ('domains_bow', feature_extraction.text.TfidfVectorizer(), 'domain')],
        remainder='passthrough')

    x_train_transformed = ct.fit_transform(x_train)
    x_test_transformed = ct.transform(x_test)

    return x_train_transformed, x_test_transformed


def build_and_verify_model(clf, x_train_tr, x_test_tr, y_train, y_test):
    """Fit model using training data and verify performance of model on test data.

    Args:
        clf: estimator/classifier. Object used to fit the training data.
        x_train_tr: sparse matrix. Training features used to train the classifier.
        x_test_tr: sparse matrix. Test features used to evaluate fitted classifier.
        y_train: Series. Training labels used to train the classifier.
        y_test: Series. Test labels used to test the classifier.

    Returns:
        accuracy: float. Accuracy score evaluated on test data.
        f1: float. F1-score evaluated on test data.
        recall: float. Recall score evaluated on test data.
        cm: numpy array. Confusion matrix evaluated on test data.
    """
    clf.fit(x_train_tr, y_train)
    y_pred = clf.predict(x_test_tr)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred, normalize='true')

    print("Accuracy :", accuracy)
    print("F1 score :", f1)
    print("Recall score :", recall)
    print("Confusion Matrix : ", cm)

    return accuracy, f1, recall, cm


def run_and_evaluate_training(con):
    """Comparative analysis of three types of classifiers.

    Compare accuracy, F1-score, recall score, and confusion matrix for
    Logistic Regression, XGBoost, and LightGBM Classifiers.
    """
    df = preprocess_valid_data(con)
    x_train, y_train, x_test, y_test = create_train_test_split(df, random_sample=False)
    x_train_transformed, x_test_transformed = transform_text_columns(x_train, x_test)

    print("Logistic Regression: ")
    clf1 = linear_model.LogisticRegression(max_iter=4000)
    accuracy, f1, recall, cm = build_and_verify_model(clf1, x_train_transformed, x_test_transformed, y_train, y_test)

    print("\nXGBoost : ")
    clf2 = XGBClassifier(booster='gbtree', max_depth=12, learning_rate=0.1, n_estimators=1000, use_label_encoder=False)
    accuracy, f1, recall, cm = build_and_verify_model(clf2, x_train_transformed, x_test_transformed, y_train, y_test)

    print("\nLightGBM : ")
    class_wts = {0: 1, 1: 1000}
    clf3 = LGBMClassifier(boosting_type='gbdt', num_leaves=5, max_depth=12, min_child_samples=20, learning_rate=0.1,
                          n_estimators=700, class_weight=class_wts)
    accuracy, f1, recall, cm = build_and_verify_model(clf3, x_train_transformed, x_test_transformed, y_train, y_test)


