import collections
from datetime import datetime
import time
import os
from urllib.parse import urlparse

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import feature_extraction, metrics
from wordcloud import WordCloud

from redbot import db

DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'


def return_post_counts(con=None):
    """Generate various counts of posts in the database."

    Args:
        con: database connection

    Returns:
        total_post_cnt: int. Number of total posts in the database
        total_hot_post_cnt: int. Number of total hot posts in the database
        total_valid_post_cnt: int. Number of total posts in the database that were created over 24 hours ago
        total_hot_valid_post_cnt: int. Number of total hot posts in the database that were created over 24 hours ago

    """
    if not con:
        con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)
    cursor = con.cursor()

    sql1 = """
    SELECT COUNT(id) FROM posts
    """
    post_cnt_gen = cursor.execute(sql1)
    total_post_cnt = list(post_cnt_gen)[0][0]

    sql2 = """
    SELECT COUNT(id) FROM posts WHERE highrank24 IS NOT NULL"""
    post_cnt_gen = cursor.execute(sql2)
    total_hot_post_cnt = list(post_cnt_gen)[0][0]

    current_time = time.time()
    current_time_utc = datetime.utcfromtimestamp(current_time)
    sql3 = """
    SELECT COUNT(id) FROM posts 
    WHERE (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 24.0
    """
    post_cnt_gen = cursor.execute(sql3, (current_time_utc,))
    total_valid_post_cnt = list(post_cnt_gen)[0][0]

    sql4 = """
    SELECT COUNT(id) FROM posts 
    WHERE (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 24 AND highrank24 IS NOT NULL
    """
    post_cnt_gen = cursor.execute(sql4, (current_time_utc,))
    total_hot_valid_post_cnt = list(post_cnt_gen)[0][0]

    print(total_post_cnt, total_hot_post_cnt, total_valid_post_cnt, total_hot_valid_post_cnt)

    return total_post_cnt, total_hot_post_cnt, total_valid_post_cnt, total_hot_valid_post_cnt


def return_scores(con=None):
    """Generate various statistics for scores of valid posts in the database."

    The score of a post is the number of upvotes minus the number of downvotes.
    A valid post is a post that was created over 24 hours ago.
    A hot post is a post that appeared among the top 10 hot posts for
    that subreddit within an hour of its creation.
    A non-hot post is a post that did not appear among the top 10 hot posts for
    that subreddit within an hour of its creation.

    Args:
        con: database connection

    Returns:
        [mean_hot, min_hot, max_hot]: list. List of mean (average), minimum, and maximum scores of valid hot posts
        [mean_non_hot, min_non_hot, max_non_hot]: list. List of mean (average), minimum, and maximum scores of valid non-hot posts
    """
    if not con:
         con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)
    cursor = con.cursor()

    current_time = time.time()
    current_time_utc = datetime.utcfromtimestamp(current_time)

    sql1 = """
    SELECT AVG(score), MIN(score), MAX(score)
    FROM posts 
    WHERE highrank24 IS NOT NULL AND (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 24
    """
    score_gen = cursor.execute(sql1, (current_time_utc,))
    ll_score_gen = list(score_gen)
    mean_hot = ll_score_gen[0][0]
    min_hot = ll_score_gen[0][1]
    max_hot = ll_score_gen[0][2]

    sql2 = """
    SELECT AVG(score), MIN(score), MAX(score)
    FROM posts 
    WHERE highrank24 IS NULL AND (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 24
    """
    score_gen = cursor.execute(sql2, (current_time_utc,))
    ll_score_gen = list(score_gen)
    mean_non_hot = ll_score_gen[0][0]
    min_non_hot = ll_score_gen[0][1]
    max_non_hot = ll_score_gen[0][2]

    print([mean_hot, min_hot, max_hot], [mean_non_hot, min_non_hot, max_non_hot])

    return [mean_hot, min_hot, max_hot], [mean_non_hot, min_non_hot, max_non_hot]


def title_keywords(con=None):
    """Analyze keywords in the titles of valid posts in the database."

    Comparitively analyze the most dominant keywords in the titles of valid hot posts
    versus the valid non-hot posts using term frequencies.
    Generate and store images of wordclouds for hot posts and non-hot posts for visual comparison.

    Args:
        con: database connection
    """
    if not con:
        con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)
    cursor = con.cursor()

    current_time = time.time()
    current_time_utc = datetime.utcfromtimestamp(current_time)

    sql1 = """
    SELECT title
    FROM posts 
    WHERE highrank24 IS NOT NULL AND (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 24
    """
    title_gen = cursor.execute(sql1, (current_time_utc,))
    ll_title_gen = list(title_gen)
    hot_titles = [elem[0] for elem in ll_title_gen]
    hot_labels = [1 for elem in ll_title_gen]

    sql2 = """
    SELECT title
    FROM posts 
    WHERE highrank24 IS NULL AND (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 24
    """
    title_gen = cursor.execute(sql2, (current_time_utc,))
    ll_title_gen = list(title_gen)
    non_hot_titles = [elem[0] for elem in ll_title_gen]
    non_hot_labels = [0 for elem in ll_title_gen]

    sql3 = """
    SELECT title
    FROM posts 
    WHERE highrank24 IS NOT NULL AND (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 <= 24
    """
    title_gen = cursor.execute(sql3, (current_time_utc,))
    ll_title_gen = list(title_gen)
    trending_titles = [elem[0] for elem in ll_title_gen]

    if not hot_titles or not non_hot_titles or not trending_titles:
        return

    all_titles = hot_titles.copy()
    all_labels = hot_labels.copy()
    all_titles.extend(non_hot_titles)
    all_labels.extend(non_hot_labels)

    vec_hot = feature_extraction.text.TfidfVectorizer(stop_words=['of', 'to', 'in', 'for', 'the', 'and', 'are', 'on',
                                                                  'as', 'from', 'is', 'he', 'his', 'it', 'that', 'was', 'with', 'by'])
    raw_features = vec_hot.fit_transform(hot_titles)
    feature_names = vec_hot.get_feature_names()
    dense = raw_features.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=feature_names)
    wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(df.T.sum(axis=1))

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.imshow(wordcloud)
    ax.set_title("Hot Post Titles", fontsize=80)
    plt.axis("off")
    if not os.path.exists(os.path.expanduser("~/ml/reddit_bot")):
        os.makedirs(os.path.expanduser("~/ml/reddit_bot"))
    fig.savefig(os.path.expanduser("~/ml/reddit_bot/hot_post_titles.png"))

    vec_non_hot = feature_extraction.text.TfidfVectorizer(stop_words=['of', 'to', 'in', 'for', 'the', 'and', 'are', 'on',
                                                                      'as', 'from', 'is', 'he', 'his', 'it', 'that', 'was', 'with', 'by'])
    raw_features = vec_non_hot.fit_transform(non_hot_titles)
    feature_names = vec_non_hot.get_feature_names()
    dense = raw_features.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=feature_names)
    wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(df.T.sum(axis=1))

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.imshow(wordcloud)
    ax.set_title("Non Hot Post Titles", fontsize=80)
    plt.axis("off")
    fig.savefig(os.path.expanduser("~/ml/reddit_bot/non_hot_post_titles.png"))

    vec_trending = feature_extraction.text.TfidfVectorizer(
        stop_words=['of', 'to', 'in', 'for', 'the', 'and', 'are', 'on',
                    'as', 'from', 'is', 'he', 'his', 'it', 'that', 'was', 'with', 'by'])
    raw_features = vec_trending.fit_transform(trending_titles)
    feature_names = vec_trending.get_feature_names()
    dense = raw_features.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=feature_names)
    wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(df.T.sum(axis=1))

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.imshow(wordcloud)
    ax.set_title("Trending Post Titles", fontsize=80)
    plt.axis("off")
    fig.savefig(os.path.expanduser("~/ml/reddit_bot/trending_post_titles.png"))


def domains(con=None):
    """Identify and analyze domains in the urls of valid posts in the database."

    Comparitively analyze the most dominant web domains in the urls of valid hot posts
    versus the valid non-hot posts.
    Generate and store images of wordclouds for hot posts and non-hot posts for visual comparison.

    Args:
        con: database connection
    """
    if not con:
        con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)
    cursor = con.cursor()

    current_time = time.time()
    current_time_utc = datetime.utcfromtimestamp(current_time)

    sql1 = """
    SELECT url
    FROM posts 
    WHERE highrank24 IS NOT NULL AND (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 24
    """
    url_gen = cursor.execute(sql1, (current_time_utc,))
    ll_url_gen = list(url_gen)
    hot_urls = [urlparse(elem[0])[1] for elem in ll_url_gen]
    hot_urls_segmented = [elem.split('.')[1] if elem.split('.')[0] == 'www' else elem.split('.')[0] for elem in hot_urls]

    sql2 = """
    SELECT url
    FROM posts 
    WHERE highrank24 IS NULL AND (strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= 24
    """
    url_gen = cursor.execute(sql2, (current_time_utc,))
    ll_url_gen = list(url_gen)
    non_hot_urls = [urlparse(elem[0])[1] for elem in ll_url_gen]
    non_hot_urls_segmented = [elem.split('.')[1] if elem.split('.')[0] == 'www'
                              else elem.split('.')[0] for elem in non_hot_urls]

    if not hot_urls_segmented or not non_hot_urls_segmented:
        return

    dd_hot = collections.Counter(hot_urls_segmented)
    dd_non_hot = collections.Counter(non_hot_urls_segmented)

    wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(dd_hot)
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.title("Hot Post Domains", fontsize=80)
    plt.axis("off")
    plt.show()
    if not os.path.exists(os.path.expanduser("~/ml/reddit_bot")):
        os.makedirs(os.path.expanduser("~/ml/reddit_bot"))
    plt.savefig(os.path.expanduser("~/ml/reddit_bot/hot_post_domains.png"))

    wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(dd_non_hot)
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.title("Non Hot Post Domains", fontsize=80)
    plt.axis("off")
    plt.show()
    plt.savefig(os.path.expanduser("~/ml/reddit_bot/non_hot_post_domains.png"))


def calculate_recall(con=None, win_hr=24.0):
    if not con:
        con = db.connect_to_db(DATABASE_DEFAULT_PATH, create_if_empty=False)

    current_time = time.time()
    current_time_utc = datetime.utcfromtimestamp(current_time)

    sql = """
    SELECT id, uid, highrank24, score, prediction FROM posts 
    WHERE (prediction IS NOT NULL) AND 
    ((strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 >= ?) AND 
    ((strftime('%s', ?) - strftime('%s', [created_utc])) / 3600.0 <= ?)
    """
    pred_df = pd.read_sql(sql, con, params=[current_time_utc, win_hr, current_time_utc, win_hr + 24.0])

    hot_or_not = []
    for idx in pred_df.index:

        if pred_df.loc[idx]['highrank24'] is None or np.isnan(pred_df.loc[idx]['highrank24']):
            hot_or_not.append(0)
        else:
            hot_or_not.append(1)

    pred_df['hot_or_not'] = hot_or_not

    recall = metrics.recall_score(pred_df['hot_or_not'], pred_df['prediction'])
    accuracy = metrics.accuracy_score(pred_df['hot_or_not'], pred_df['prediction'])
    print("Time :", current_time_utc)
    print("Recall :", recall)
    print("Accuracy :", accuracy)
    print(pred_df[(pred_df['hot_or_not'] == 1) | (pred_df['prediction'] == 1)])

    return pred_df, recall, accuracy