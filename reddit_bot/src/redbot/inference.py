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


def create_transformer_and_model(df, to_save=False):
    """Generate column transformer and classifier/estimator object fitted to all of the valid data.

    Args:
        df: DataFrame.  Preprocessed DataFrame with features and labels
        obtained from the function preprocess_valid_data() of the train module.
        to_save: bool. If True, save the estimator/classifier model and column transformer objects.

    Returns:
        ct: Fitted column transformer object
        clf: Fitted estimator/classifier object
    """
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


def retrieve_hour_old_posts(con):
    """Retrieve posts ingested between 1 and 3 hours ago that have no predictions.

    Use posts that were ingested at least one hour prior to making predictions
    so that the score (upvotes minus downvotes) for those posts has a valid stable value.

    Args:
        con: database connection.

    Returns:
        raw_test_df: DataFrame. DataFrame containing raw attributes
        (uuid, url, title, score, upvote_ratio) for posts ingested
        between 1 and 3 hours ago.
    """
    current_time = time.time()
    current_time_utc = datetime.utcfromtimestamp(current_time)

    raw_test_df = db.retrieve_posts_for_inference(con, current_time_utc)

    return raw_test_df


def preprocess_hour_old_posts(raw_test_df):
    """Preprocess posts that were ingested between 1 and 3 hours ago, for which there are no predictions.

    Preprocess the raw attributes by replacing the url with the domain and dropping the primary key.
    Generate a prediction table containing the primary keys of the posts.

    Args:
        raw_test_df: DataFrame containing raw attributes
        (uuid, url, title, score, upvote_ratio) for posts ingested
        between 1 and 3 hours ago.

    Returns:
        df: DataFrame. DataFrame containing features (domains, titles, scores, upvote_ratios)
        of the hour old posts.
        prediction_table: DataFrame. DataFrame containing ids (primary keys)
        of the hour old posts.
    """
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
    prediction_table['uuid'] = df['uuid'].copy()

    df.drop(['url', 'uuid'], axis=1, inplace=True)

    return df, prediction_table


def save_predictions_to_table(prediction_table, con):
    """Update the prediction made for the hour old posts to posts table in the database.

    Args:
        prediction_table: DataFrame. Contains the primary key and prediction for the hour old posts.
        con: database connection.
    """
    for idx in prediction_table.index:
        uuid = prediction_table.loc[idx]['uuid']
        pred = prediction_table.loc[idx]['predictions']
        db.update_prediction(con, pred, uuid)


def run_inference(con, new_model=True, to_save=False):
    """Function to run inference on hour old posts.

    Generate new model and column transformer objects or retrieve saved objects.
    Transform text features of hour old posts to sparse numerical matrices using the column transformer object.
    Run prediction on the generated matrix. Modify the prediction table to add a column of prediction values, one
    for each post corresponding to each primary key.

    Args:
        con: database connection.
        new_model: bool. If True, generate new estimator/classifier and column transformer object.
        If False, retrieve saved objects.
        to_save: bool. If True, save generated estimator/classifier and column transformer objects.

    Returns:
        prediction_table: DataFrame. Contains a column of primary keys ("id") and a column of prediction values
        ("predictions") for each retrieved hour old post.
    """
    if new_model:
        df = train.preprocess_valid_data()
        ct, clf = create_transformer_and_model(df, to_save)
    else:
        clf = load(open(os.path.expanduser("~/github/ds/reddit_bot/model_objects/model.pkl"), 'rb'))
        ct = load(open(os.path.expanduser("~/github/ds/reddit_bot/model_objects/ct.pkl"), 'rb'))

    raw_test_df = retrieve_hour_old_posts(con)

    df_test, prediction_table = preprocess_hour_old_posts(raw_test_df)
    if len(df_test.index) > 0:
        x_test_transformed = ct.transform(df_test)
        y_pred = clf.predict(x_test_transformed)
    else:
        y_pred = []

    prediction_table['predictions'] = y_pred

    save_predictions_to_table(prediction_table, con)

    return prediction_table


