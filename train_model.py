import pickle
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def train_and_evaluate_model(data_frame, model_name):
    print("Transforming Data for tfidf")
    x_train, x_test, y_train, y_test = transform_data(data_frame)
    print(f"Fitting {model_name} model")
    model = get_model(model_name)
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    print('Evaluating Model')
    evaluate_model(model, model_name, x_test, y_test)
    print('Plotting roc auc curve')
    plot_roc_auc(model_name, y_test, y_predicted)
    return model


def get_model(model_name):
    if model_name == 'bernoulli':
        return BernoulliNB()
    elif model_name == 'svc':
        return LinearSVC()
    elif model_name == 'logistic_regression':
        return LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    else:
        raise Exception('Invalid model chosen. Select one of ("svc", "bernoulli", "logistic_regression")')


def transform_data(df):
    x_train, x_test, y_train, y_test = train_test_split(df.Text, df.Sentiment, test_size=.1, stratify=df.Sentiment)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectorizer.fit(x_train)
    # print('No. of feature_words: ', len(vectorizer.get_feature_names()))
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    pickle.dump(vectorizer, open(f'trained_models/trained_vectorizer_model.pk', 'wb'))
    return x_train, x_test, y_train, y_test


def evaluate_model(model, model_name, x_test, y_test):
    # Predict values for Test dataset
    y_predicted = model.predict(x_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_predicted))

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_predicted)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title(f"Confusion Matrix for {model_name}", fontdict={'size': 18}, pad=20)


def plot_roc_auc(model_name, y_test, y_predicted):
    fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC CURVE for {model_name}')
    plt.legend(loc="lower right")
    plt.show()
