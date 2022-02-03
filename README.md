# Twitter Sentiment Analysis
* Train Model On Twitter Data
* Access Model Accuracy
* Predict live tweets

# Data Set preparation
* Dataset can be downloaded from https://www.kaggle.com/kazanova/sentiment140?fbclid=IwAR1iiExE2tA3qsQU3rkgzMmMQ2AJUL5nvaehOjBMhq6VTbE29TnQYRwy2oY

* Store downloaded inside dataset directory with filename as sentiment_data.csv i.e dataset/sentiment_data.csv

* Run Command Below to download stop words
```
import nltk
nltk.download('stopwords')
ntlk.download('wordnet')
nltk.download('omw-1.4')
```

* wordcloud module has dependency on microsoft tools