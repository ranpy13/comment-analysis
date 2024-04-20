import os
import pandas as pd
import tensorflow as tf
import numpy as np
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, hamming_loss, fbeta_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, ShuffleSplit, learning_curve
from statistics import mean
from wordcloud import WordCloud
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
test_y = pd.read_csv("data/test_labels.csv")

print(train.head(), test.head(), test_y.head())
print(train.describe())
print(test.shape, train.shape)


sns.set_theme(color_codes= True)
comment_len = train.comment_text.str.len()
sns.histplot(comment_len, kde= False, bins= 20, color='steelblue')


# subsetting labels from the training data
train_labels = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
label_count = train_labels.sum()

plt.plot(label_count, kind = 'bar', title= 'Labels Frequency', color= 'Steelblue')


# code to draw bar graph for visualisign distribution of classes within each label
barWidth = 0.25
bars1 = [sum(train['toxic'] == 1), sum(train['obscene'] == 1), sum(train['insult'] == 1), sum(train['severe_toxic'] == 1),
         sum(train['identity_hate'] == 1), sum(train['threat'] == 1)]
bars2 = [sum(train['toxic'] == 0), sum(train['obscene'] == 0), sum(train['insult'] == 0), sum(train['severe_toxic'] == 0),
         sum(train['identity_hate'] == 0), sum(train['threat'] == 0)]

r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars1, color= 'steelblue', width= barWidth, label= 'labeled = 1')
plt.bar(r2, bars2, color= 'lightsteelblue', width= barWidth, label= 'labeled = 0')

plt.xlabel('group', fontweight= 'bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Toxic', 'Obscene', 'Insult', 'Severe Toxic', 'Identity Hate', 'Threat'])
plt.legend()
plt.show()

# example of clean comment
print(train.comment_text[0])
# example of toxic comment
print(train[train.toxic == 1].iloc[1, 1])


# cross correlation matrix across labels
rowsums = train.iloc[:, 2:].sum(axis= 1)
temp = train.iloc[:, 2:-1]
train_corr = temp[rowsums > 0]
corr = train_corr.corr()
plt.figure(figsize= (10, 8))
sns.heatmap(corr, xticklabels= corr.columns.values, yticklabels= corr.columns.values, annot= True, cmap= 'Blues')


# generating word clouds
# vizualize the most common words conitrbuting to the token
def W_Cloud(token):
    threat_context = train[train[token] == 1]
    threat_text = threat_context.comment_text
    neg_text = pd.Series(threat_text).str.cat(sep=' ')
    wordcloud = WordCloud(width= 1600, height= 800, max_font_size= 200).generate(neg_text)

    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud.recolor(colormap= "Blues"), interpolation= 'bilinear')
    plt.axis('off')
    plt.title(f"Most cmmon words associated with {token} comment", size= 20)
    plt.show()

token = input("Choose a class to vizualize the most common words contributing to the class: ")
W_Cloud(token.lower())



## Feature Engineering


