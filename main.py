import os
import pandas as pd
import tensorflow as tf
import numpy as np
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from timeit import default_timer as timer

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
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
test_labels = ["toxic", "severe_toxic", "obsence", "threat", "insult", "identify_hate"]
def tokenize(text):
    '''
    Tokenize text and return a non-unique list of tokenized words found in the text.
    Normalize to lowercase, strip punctuation, remove stop words, filter non-ascii characters.
    Lemmatize the words and lastly drop words of length < 3. 
    '''
    text = text.lower()
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    words = nopunct.split(' ')

    #remove any non ascii
    words = [word.encode('ascii', 'ignore').decode('ascii') for word in words]
    lmtzr = [lmtzr.lemmatize(w) for w in words]
    words = [w for w in words if len(w) > 2]
    return words



# benchmarking different vectors
vector = TfidfVectorizer(ngram_range=(1, 1), analyzer= 'word', tokenizer= tokenize, stop_words= 'english', strip_accents= 'unicode', use_idf= 1, min_df= 10)
X_train = vector.fit_transform(train['comment_text'])
X_test = vector.transform(test['comment_text'])

print(vector.get_feature_names()[0:20])



# Modeling and Evalution
# baseline model - naive bayes

# cross-validation
clf1 = MultinomialNB()
clf2 = LogisticRegression()
clf3 = LinearSVC()

def cross_validation_score(classifier, X_train, y_train):
    methods = []
    name = classifier.__class__.__name__.split('.')[-1]

    for label in test_labels:
        recall = cross_val_score(classifier, X_train, y_train[label], cv= 10, scoring= 'recall')
        f1 = cross_val_score(classifier, X_train, y_train[label], cv= 10, scoring= 'f1k')
        methods.append([name, label, recall.mean(), f1.mean()])
    
    return methods


methods1_cv = pd.DataFrame(cross_validation_score(clf1, X_train, train))
methods2_cv = pd.DataFrame(cross_validation_score(clf2, X_train, train))
methods3_cv = pd.DataFrame(cross_validation_score(clf3, X_train, train))

# creating a dataframe to show summary of results
methods_cv = pd.concat([methods1_cv, methods2_cv, methods3_cv])
methods_cv.columns = ['Model', 'Label', 'Recall', 'F1']
meth_cv = methods_cv.reset_index()
print(meth_cv[['Model', 'Label', 'Recall', 'F1']])





# X modeling and evaluation
def score(classifier, X_train, y_train, X_test, y_test):
    methods = []
    hloss = []
    name = classifier.__class__.__name__.split('.')[-1]
    predict_df = pd.DataFrame()
    predict_df['id'] = test_y['id']

    for label in test_labels:
        classifier.fit(X_train, y_train[label])
        predicted = classifier.predict(X_test)
        predict_df[label] = predicted
        
        recall = recall_score(y_test[y_test[label] != -1][label], predicted[y_test[label] != -1], average= 'weighted')
        f1 = f1_score(y_test[y_test[label] != -1][label], predicted[y_test[label] != -1], average= 'weighted')
        conf_mat = confusion_matrix(y_test[y_test[label] != -1][label], predicted[y_test[label] != -1])
    
    hamming_loss_score = hamming_loss(test_y[test_y['toxic'] != -1].iloc[:, 1:7], predict_df[test_y['toxic'] != -1].iloc[:, 1:7])
    hloss.append([name, hamming_loss_score])

    return hloss, methods


h1, method1 = score(clf1, X_train, train, X_test, test_y)
h2, method2 = score(clf2, X_train, train, X_test, test_y)
h3, method3 = score(clf3, train, X_test, test_y)


# summary of results
methods1 = pd.DataFrame(method1)
methods2 = pd.DataFrame(method2)
methods3 = pd.DataFrame(method3)

methods = pd.concat([methods1, methods2, methods3])
methods.columns = ['Model', 'Label', 'Recall', 'F1', 'Confusion Matrix']
meth = methods.reset_index()
print(meth[['Model', 'Label', 'Recall', 'F1']])


# Visualizing F1 score results through box-plot
ax = sns.boxplot(x= 'Model', y= 'F1', data= methods, palette= 'Blues')
sns.stripplot(x= 'Model', y= 'F1', data= methods, size= 8, edgecolor= 'gray', linewidth= 2, palette= 'Blues')
ax.set_xticklabels(ax.get_xticklabels(), rotation= 20)
plt.show()



# Vizualizing performance per classifier accross each category
print("Plot for Multinomial Naive Bayes Classifier")
m2 = methods[methods.Model =='MultinomialNB']
m2.set_index(["Label"], inplace= True)
m2.plot(figsize= (16, 8), kind='bar', title= 'Metrics', rot= 60, ylim= (0.0, 1), colormap= 'tab10')


# for Logistic regression
m2 = methods[methods.Model =='LogisticRegression']
m2.set_index(["Label"], inplace= True)
m2.plot(figsize= (16, 8), kind= 'bar', title= 'Metrics', rot= 60, ylim= (0.0, 1), colormap= 'tab10')


# for Linear SVC
m2 = methods[methods.Model =='LinearSVC']
m2.set_index(["Label"], inplace= True)
m2.plot(figsize= (16, 8), kind= 'bar', title= 'Metrics', rot= 60, ylim= (0.0, 1), colormap= 'tab10')


# Confusion matrix visualization
def drawConfuisonMatrix(cm):
    cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]
    ax = plt.axes()
    sns.heatmap(cm, 
                annot= True,
                annot_kws= {'size': 16},
                fmt= '.2f',
                linewidths=2, 
                linecolor= 'steelblue',
                xticklabels= ('Non-toxic', 'Toxic'),
                yticklabels= ('Non-toxic', 'Toxic'))
    
    plt.ylabel('True', fontsize= 18)
    plt.xlabel('Predicted', fontsize= 18)
    plt.show()


def Matrix(label):
    print(f"************{label} labelling**************")
    labels = {"toxic": 0, "severe_toxic": 1, "obscene": 2,
              "threat": 3, "insult": 4, "identity_hate": 5}
    
    pos = labels[label]
    for i in ragne(pos, len(meth), 6):
        print()
        print(f"******** {meth['Model'][i]} ***********")
        cm = meth['Confusion_Matrix'][i]
        drawConfuisonMatrix(cm)


token = input("Choose a class for the Confusion Matrix: ")
Matrix(token.lower())



# Aggregated Hamming Loss Score
hl1_df = pd.DataFrame(h1)
hl2_df = pd.DataFrame(h2)
hl3_df = pd.DataFrame(h3)

hammingLoss = pd.concat([hl1_df, hl2_df, hl3_df])
hammingLoss.columns = ['Model', 'Hamming Loss']
h1 = hammingLoss.reset_index()
print(h1[['Model', 'Hamming_Loss']])


# Pipelines
pipe_lr = Pipeline([('lr', LogisticRegression(class_weight= 'balanced'))])
pipe_linear_svm = Pipeline(['svm', LinearSVC(class_weight= {1: 20})])
pipelines = [pipe_lr, pipe_linear_svm]

score_df = []
for pipe in pipelines:
    f1_values = []
    recall_values = []
    hl = []
    training_time = []
    predict_df = []
    predict_df['id'] = test_y['id']
    for label in test_labels:
        start = timer()
        pipe.fit(X_train, train[label])
        train_time = timer() - start
        predicted = pipe.predict(X_test)
        predict_df[label] = predicted

        f1_values.append(f1_score(test_y[test_y[label] != -1][label], predicted[test_y[label] != -1], average= 'weighted'))
        recall_values.append(recall_score(test_y[test_y[label] != -1][label], predicted[test_y[label] != -1], average= 'weighted'))
        training_time.append(train_time)
        name = pipe.steps[-1][1].__class__.__name__.split('.')[-1]

    hamming_loss_score = hamming_loss(test_y[test_y['toxic'] != -1].iloc[:, 1:7], predict_df[test_y['toxic'] != -1].iloc[:, 1:7])
    val = [name, mean(f1_values, mean(recall_values), hamming_loss_score, mean(training_time))]
    score_df.append(val)

scores = pd.DataFrame(score_df)
scores.columns = ['Model', 'F1', 'Recall', 'Hamming Loss', 'Training_Time']
print(scores)




# Hyperparameter tuning with grid search CV
logistic_regression_classifier = LogisticRegression()
parameter_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'], 'class_weight': [None, 'balanced']}
cross_validation = StratifiedKFold(n_splits= 5)

grid_search = GridSearchCV(logistic_regression_classifier, param_grid= parameter_grid, cv= cross_validation, scoring= 'f1')
grid_search.fit(X_train, train['toxic'])

print("Best parameters: {}".format(grid_search.best_params_))
print("Best estimater: {}".format(grid_search.best_estimator_))



# SVM Classifier Tuning
svm_classifier = LinearSVC()
parameter_grid = {'class_weight': [None, 'balanced'], 'C': [1, 5, 10]}
cross_validation = StratifiedKFold(n_splits= 5)
grid_search = GridSearchCV(svm_classifier, param_grid= parameter_grid, cv= cross_validation, scoring= 'f1')

grid_search.fit(X_train, train['toxic'])
print("Best parameters: {}".format(grid_search.best_params_))
print("Best estimator: {}".format(grid_search.best_estimator_))




# Model Selection
svm_clf = LinearSVC(c= 1, class_weight= None, dual= True, fit_intercept= True, intercept_scaling= 1, loss= 'squared_hinge', max_iter= 1000, multi_class= 'ovr', penalty= 'l2', random_state= None, tol= 0.0001, verbose= 0)
lr_clf = LogisticRegression(c= 1.0, class_weight= None, dual= False, fit_intercept= True, intercept_scaling= 1, max_iter= 100, multi_class= 'warn', n_jobs= None, penalty= 'l2', random_state= None, solver= 'lbfgs', tol= 0.0001, verbose= 0, warm_start= False)
tunned_model_score_df = []
for model in [svm_clf, lr_clf]:
    f1_values = []
    recall_values = []
    hl = []
    training_time = []
    predict_df = pd.DataFrame()
    predict_df['id'] = test_y['id']

    for label in test_labels:
        start = timer()
        model.fit(X_train, train[label])
        training_time.append(timer() - start)
        predicted = model.predict(X_test)
        predict_df[label] = predicted

        f1_values.append(f1_score(test_y[test_y[label] != -1][label], predicted[test_y[label] != -1], average="weighted"))
        recall_values.append(recall_score(test_y[test_y[label] != -1][label], predicted[test_y[label] != -1], average="weighted"))
        name = model.__class__.__name__

    hamming_loss_score = hamming_loss(test_y[test_y['toxic'] != -1].iloc[:, 1:7], predict_df[test_y['toxic'] != -1].iloc[:, 1:7])
    val = [name, mean(f1_values), mean(recall_values), hamming_loss_score, sum(training_time)]

    print("Tunned Model Score = ", tunned_model_score_df.append(val))


tunned_scores = pd.DataFrame(tunned_model_score_df,)
tunned_scores.columns = ['Model', 'F1',
                         'Recall', 'Hamming_Loss', 'Traing_Time']
print(tunned_scores)





# Ensembling
# boosting models

ab_clf = AdaBoostClassifier()
gb_clf = GradientBoostingClassifier()
xg_clf = xgb.XGBClassifier()
boosting_models = [ab_clf, gb_clf, xg_clf]

boosting_score_df = []
for model in boosting_models:
    f1_values = []
    recall_values = []
    training_time = []
    hloss = []
    predict_df = pd.DataFrame()
    predict_df['id'] = test_y['id']

    for idx, label in enumerate(test_labels):
        start = timer()
        model.fit(X_train, train[label])
        predicted = model.predict(X_test)
        training_time.append(timer() - start)
        predict_df[label] = predicted
        f1_values.append(f1_score(test_y[test_y[label] != -1][label], predicted[test_y[label] != -1], average="weighted"))
        recall_values.append(recall_score(test_y[test_y[label] != -1][label], predicted[test_y[label] != -1], average="weighted"))
        name = model.__class__.__name__
    
    hamming_loss_score = hamming_loss(test_y[test_y['toxic'] != -1].iloc[:, 1:7], predict_df[test_y['toxic'] != -1].iloc[:, 1:7])
    val = [name, mean(f1_values), mean(recall_values), hamming_loss_score, mean(training_time)]
    boosting_score_df.append(val)


# scores after boosting the models
boosting_score = pd.DataFrame(boosting_score_df,)
boosting_score.columns = ['Model', 'F1', 'Recall', 'Hamming_Loss', 'Training_Time']
print("Boosting Score: \n", boosting_score)


# Voting Classifier
ensemble_clf = VotingClassifier(estimators=[('lr', lr_clf),
                                            ('svm', svm_clf),
                                            ('xgb', xgb_clf)], voting='hard')
ensemble_score_df = []
f1_values = []
recall_values = []
hl = []
training_time = []

predict_df = pd.DataFrame()
predict_df['id'] = test_y['id']
for label in test_labels:
    start = timer()
    ensemble_clf.fit(X_train, train[label])
    training_time.append(timer() - start)
    predicted = ensemble_clf.predict(X_test)
    predict_df[label] = predicted
    f1_values.append(f1_score(test_y[test_y[label] != -1][label],
                              predicted[test_y[label] != -1],
                              average="weighted"))
    recall_values.append(recall_score(test_y[test_y[label] != -1][label],
                                      predicted[test_y[label] != -1],
                                      average="weighted"))
    name = 'Ensemble'

hamming_loss_score = hamming_loss(test_y[test_y['toxic'] != -1].iloc[:, 1:7],
                                  predict_df[test_y['toxic'] != -1].iloc[:, 1:7])

val = [name, mean(f1_values), mean(recall_values),
       hamming_loss_score, mean(training_time)]
ensemble_score_df.append(val)


# printing the values
ensemble_score = pd.DataFrame(ensemble_score_df,)
ensemble_score.columns = ['Model', 'F1',
                          'Recall', 'Hamming_Loss', 'Training_Time']
print("Ensemble Score \n", ensemble_score)




# result interpretation
label = 'toxic'
lr = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
               intercept_scaling=1, loss='squared_hinge', max_iter=1000,
               multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
               verbose=0)
lr.fit(X_train, train[label])
Toxic_LR = lr.predict(X_test)
test_combined = pd.concat([test, test_y], axis=1)

commentCheck = test_combined[(test_combined.toxic == 1) & (
    Toxic_LR == 0)].comment_text
print("CommentCheck.shape = ", commentCheck.shape)



# extract wrongly classified comments
commentCheck = test_combined[(test_combined.toxic == 1) & (
    Toxic_LR == 0)].comment_text

neg_Check = pd.Series(commentCheck).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,
                      max_font_size=200).generate(neg_Check)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud.recolor(colormap="Blues"), interpolation='bilinear')
plt.axis("off")
plt.title("Most common words from misclassified", size=20)
plt.show()


wrongWords = tokenize(neg_Check)
stop_words = stopwords.words('English')
wrongWords = [w for w in wrongWords if w not in stop_words]
cntr = Counter(wrongWords)
cntr.most_common(20)

neg_text_train = train['comment_text'].str.cat(sep=' ')
cntr_train = Counter(tokenize(neg_text_train))
cntr_train.get('ucking')

