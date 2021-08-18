from datetime import datetime
import os
import time
from urllib.parse import urlparse

from lightgbm import LGBMClassifier
import pandas as pd
from pickle import dump, load
from sklearn import compose, feature_extraction

from redbot import db
from redbot import train


DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'


def create_transformer_and_model(df, to_save=False):
    y_train = df['hot_or_not']
    x_train = df.drop(['hot_or_not'], axis=1)

    ct = compose.ColumnTransformer(
        [('titles_bow',
          feature_extraction.text.TfidfVectorizer(stop_words=['of', 'to', 'in', 'for', 'the', 'and', 'are', 'on',
                                                              'as', 'from', 'is', 'he', 'his', 'it', 'that', 'was',
                                                              'with', 'by']), 'title'),
         ('domains_bow', feature_extraction.text.TfidfVectorizer(), 'domain')],
        remainder='passthrough')

    x_train_transformed = ct.fit_transform(x_train)

    class_wts = {0: 1, 1: 1000}
    clf = LGBMClassifier(boosting_type='gbdt', num_leaves=5, max_depth=12, min_child_samples=20, learning_rate=0.1,
                         n_estimators=700, class_weight=class_wts)
    clf.fit(x_train_transformed, y_train)

    if to_save:
        if not os.path.exists(os.path.expanduser("~/github/ds/reddit_bot/model_objects")):
            os.makedirs(os.path.expanduser("~/github/ds/reddit_bot/model_objects"))
        dump(ct, open(os.path.expanduser("~/github/ds/reddit_bot/model_objects/ct.pkl"), "wb"))
        dump(clf, open(os.path.expanduser("~/github/ds/reddit_bot/model_objects/model.pkl"), "wb"))

    return ct, clf


def retrieve_hour_old_posts(con=None):
    if not con:
        con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)

    current_time = time.time()
    current_time_utc = datetime.utcfromtimestamp(current_time)
    sql = """
    SELECT id, url, title, score, upvote_ratio FROM posts 
    WHERE (prediction IS NULL) AND  
    ((strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 1.0) AND 
    ((strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 <= 3.0)
    """
    raw_test_df = pd.read_sql(sql, con, params=[current_time_utc, current_time_utc])

    return raw_test_df


def preprocess_hour_old_posts(raw_test_df):
    df = raw_test_df.copy()
    full_domain = []
    for idx in df.index:
        domain = urlparse(df.loc[idx]['url'])[1]
        if domain.split('.')[0] == 'www':
            full_domain.append(domain.split('.')[1])
        else:
            full_domain.append(domain.split('.')[0])

    df['domain'] = full_domain
    prediction_table = pd.DataFrame()
    prediction_table['id'] = df['id'].copy()

    df.drop(['url', 'id'], axis=1, inplace=True)

    return df, prediction_table


def save_predictions_to_table(prediction_table, con=None):
    if not con:
        con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)

    for idx in prediction_table.index:
        idy = prediction_table.loc[idx]['id']
        pred = prediction_table.loc[idx]['predictions']
        db.update_prediction(con, pred, idy)


def run_inference(new_model=True, to_save=False):
    if new_model:
        df = train.preprocess_valid_data()
        ct, clf = create_transformer_and_model(df, to_save)
    else:
        clf = load(open(os.path.expanduser("~/github/ds/reddit_bot/model_objects/model.pkl"), 'rb'))
        ct = load(open(os.path.expanduser("~/github/ds/reddit_bot/model_objects/ct.pkl"), 'rb'))

    raw_test_df = retrieve_hour_old_posts()

    df_test, prediction_table = preprocess_hour_old_posts(raw_test_df)
    if len(df_test.index) > 0:
        x_test_transformed = ct.transform(df_test)
        y_pred = clf.predict(x_test_transformed)
    else:
        y_pred = []

    prediction_table['predictions'] = y_pred

    save_predictions_to_table(prediction_table)

    return prediction_table


