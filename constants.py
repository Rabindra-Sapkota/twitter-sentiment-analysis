TWITTER_DATASET = 'dataset/sentiment_data.csv'
TWITTER_DATA_HEADER = ['Sentiment', 'ids', 'date', 'flag', 'user', 'Text']

VALID_MODELS = ['svc', 'bernoulli', 'logistic_regression']
MODEL_TO_TRAIN = 'logistic_regression'

# 'Predict' or 'Train'
RUN_METHOD = 'Predict'

TEST_TWEET_TO_PREDICT = ['I am very happy', 'I am sad', 'I am depressed', 'I am good', 'I am in joy']
SENTIMENT_DICT = {0: "Negative", 1: "Positive"}

# Twitter Credentials
ACCESS_TOKEN = ''
ACCESS_TOKEN_SECRET = ''
CONSUMER_KEY = ''
CONSUMER_SECRET = ''
TWEET_TEXT_QUERY = 'Covid'
TWEET_EXTRACTION_COUNT = 10
