from datetime import datetime
import time
from urllib.parse import urlparse

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn import compose, feature_extraction, linear_model, metrics, model_selection
import sqlite3
from xgboost import XGBClassifier

from redbot import db


DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'


def preprocess_valid_data(con=None):
    if not con:
        con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)
    cursor = con.cursor()

    current_time = time.time()
    current_time_utc = datetime.utcfromtimestamp(current_time)
    sql = """
    SELECT url, title, score, upvote_ratio, highrank24 FROM posts 
    WHERE (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 24.0
    """
    raw_df = pd.read_sql(sql, con, params=[current_time_utc])

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
    clf.fit(x_train_tr, y_train)
    y_pred = clf.predict(x_test_tr)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred, normalize='true')

    print("Accuracy :", accuracy)
    print("F1 score :", f1)
    print("Confusion Matrix : ", cm)

    return accuracy, f1, cm


def run_and_evaluate_training():
    df = preprocess_valid_data()
    x_train, y_train, x_test, y_test = create_train_test_split(df, random_sample=True)
    x_train_transformed, x_test_transformed = transform_text_columns(x_train, x_test)

    print("Logistic Regression: ")
    clf1 = linear_model.LogisticRegression(max_iter=4000)
    accuracy, f1, cm = build_and_verify_model(clf1, x_train_transformed, x_test_transformed, y_train, y_test)

    print("\nXGBoost : ")
    class_wts = {0: 1, 1: 1000}
    clf2 = XGBClassifier(booster='gbtree', max_depth=12, learning_rate=0.1, n_estimators=1000, use_label_encoder=False)
    accuracy, f1, cm = build_and_verify_model(clf2, x_train_transformed, x_test_transformed, y_train, y_test)

    print("\nLightGBM : ")
    class_wts = {0: 1, 1: 1000}
    clf3 = LGBMClassifier(boosting_type='gbdt', num_leaves=5, max_depth=12, min_child_samples=20, learning_rate=0.1,
                          n_estimators=700, class_weight=class_wts)
    accuracy, f1, cm = build_and_verify_model(clf3, x_train_transformed, x_test_transformed, y_train, y_test)


