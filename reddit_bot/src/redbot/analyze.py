import collections
from datetime import datetime
import time
import os
from urllib.parse import urlparse

from matplotlib import pyplot as plt
import pandas as pd
from sklearn import feature_extraction, feature_selection
from wordcloud import WordCloud

from redbot import db

DATABASE_DEFAULT_PATH = '~/.redbot/redbotdb.sqlite'


def return_post_counts(con=None):
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

    all_titles = hot_titles.copy()
    all_labels = hot_labels.copy()
    all_titles.extend(non_hot_titles)
    all_labels.extend(non_hot_labels)

    vec_hot = feature_extraction.text.TfidfVectorizer(stop_words=['of', 'to', 'in', 'for', 'the', 'and', 'are', 'on', 'as', 'from'])
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
    fig.savefig(os.path.expanduser(f"~/ml/reddit_bot/hot_post_titles.png"))

    vec_non_hot = feature_extraction.text.TfidfVectorizer(stop_words=['of', 'to', 'in', 'for', 'the', 'and', 'are', 'on', 'as', 'from'])
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
    fig.savefig(os.path.expanduser(f"~/ml/reddit_bot/non_hot_post_titles.png"))


def domains(con=None):
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
    hot_labels = [1 for elem in ll_url_gen]

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
    non_hot_labels = [0 for elem in ll_url_gen]

    dd_hot = collections.Counter(hot_urls_segmented)
    dd_non_hot = collections.Counter(non_hot_urls_segmented)

    wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(dd_hot)
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.title("Hot Post Domains", fontsize=80)
    plt.axis("off")
    plt.show()
    plt.savefig(os.path.expanduser("~/ml/reddit_bot/hot_post_domains.png"))

    wordcloud = WordCloud(width=1000, height=500).generate_from_frequencies(dd_non_hot)
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.title("Non Hot Post Domains", fontsize=80)
    plt.axis("off")
    plt.show()
    plt.savefig(os.path.expanduser("~/ml/reddit_bot/non_hot_post_domains.png"))

#total_post_cnt, total_hot_post_cnt, total_valid_post_cnt, total_hot_valid_post_cnt = return_post_counts()

#[mean_hot, min_hot, max_hot], [mean_non_hot, min_non_hot, max_non_hot] = return_scores()

#title_keywords_hot_posts()