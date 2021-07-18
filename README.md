My solutions to problems posted on aicrowd, kaggle, and analysis done on UCI datasets.

## Sensor Signal Time Series Analysis
* [Predict](https://www.kaggle.com/c/predict-volcanic-eruptions-ingv-oe) time to eruption of volcanoes based on sensor signal data
* [Classify](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#) activity of person wearing wearable IMU as walking, walking upstairs, walking
downstairs, sitting, standing, or laying

Used time domain signal representations of the sensor signals as inputs to 1D convolutional and
bidirectional LSTM layers, and provided [frequency domain representations](https://github.com/keyapandia123/ds/blob/main/volcano/ingv.ipynb) (2D spectrograms) and [wavelet spectra](https://github.com/keyapandia123/ds/blob/main/activity_recog/activity_recog.ipynb) of the sensor signals as input to 2D convolutional layers . Placed 5th out of 620 teams on the volcanic prediction competition

## Computer Vision (Classification)
* [Given](https://www.aicrowd.com/challenges/ai-blitz-6) an image of a chess board, predict piece count, points, board configuration, and side likely to win. Given a video sequence of a game, identify the moves being made
* [Given](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) chest x-ray images, classify patient status as normal vs pneumonia

[Implemented](https://github.com/keyapandia123/ds/tree/main/chess) classical image processing techniques (intensity thresholding, filtering / blurring, 2D correlation, template matching), tree-based techniques (XGBoost), and deep learning techniques (CONV2D in Keras) with and without [transfer learning](https://github.com/keyapandia123/ds/blob/main/chest_xray/chest_xray.ipynb)

## Tabular Data (Regression, Classification, and Analytics)
* [Given](https://www.aicrowd.com/challenges/ai-blitz-4/problems/crdio) tabular data comprising fetal cardiotocogram (CTG) features, predict risk to fetus
* [Analyze](https://www.kaggle.com/c/kaggle-survey-2020) survey response data about the state of the machine learning / data science community

Explored correlation matrices, statistical separation using [t-tests](https://github.com/keyapandia123/ds/blob/main/cardio/cardio.ipynb), seaborn and matplotlib plotting tools, [geopandas](https://github.com/keyapandia123/ds/blob/main/ml_ds_survey/summary_2020.ipynb), and compared classifiers and regressors

## Natural Language Processing
* [Given](https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set) sentences from user-provided reviews of products or services, classify user-sentiments as positive or negative
* [Given](https://www.kaggle.com/hgultekin/bbcnewsarchive) text content and headlines from news articles, predict the topic/category of news as
entertainment, business, politics, sports, or tech

Investigated sklearn vectorizers, t-SNE, clustering, chi2 correlations, [wordclouds](https://github.com/keyapandia123/ds/blob/main/bbc/bbc.ipynb), keras tokenizer and [embeddings](https://github.com/keyapandia123/ds/blob/main/sentiments/sentiments.ipynb)

