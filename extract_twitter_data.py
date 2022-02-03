import time
import constants
import tweepy

auth = tweepy.OAuth1UserHandler(consumer_key=constants.CONSUMER_KEY, consumer_secret=constants.CONSUMER_SECRET)
auth.set_access_token(key=constants.ACCESS_TOKEN, secret=constants.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth=auth, wait_on_rate_limit=True)

try:
    # Creation of query method using parameters
    tweets = tweepy.Cursor(api.search_tweets, q=constants.TWEET_TEXT_QUERY).items(constants.TWEET_EXTRACTION_COUNT)

    # Pulling information from tweets iterable object
    tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]

    # Creation of dataframe from tweets list
    # Add or remove columns as you remove tweet information
    # tweets_df = pd.DataFrame(tweets_list)
    print(tweets_list)

except BaseException as e:
    print('failed on_status,', str(e))
    time.sleep(3)
