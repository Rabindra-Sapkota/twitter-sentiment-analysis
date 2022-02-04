from wordcloud import WordCloud
import matplotlib.pyplot as plt


def negative_word_plot(df):
    data_neg = df[df.Sentiment == 0]['Sentiment']
    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=1000, width=1600, height=800,
                   collocations=False).generate(" ".join(data_neg))
    plt.imshow(wc)


def positive_word_plot(df):
    data_pos = df[df.Sentiment == 1]['Sentiment']
    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=1000, width=1600, height=800,
                   collocations=False).generate(" ".join(data_pos))
    plt.imshow(wc)
