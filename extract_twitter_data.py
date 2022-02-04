import re
import time
import pandas as pd
import constants
import tweepy


def extract_tweets(tweet_text, tweet_count):
    auth = tweepy.OAuth1UserHandler(consumer_key=constants.CONSUMER_KEY, consumer_secret=constants.CONSUMER_SECRET)
    auth.set_access_token(key=constants.ACCESS_TOKEN, secret=constants.ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth=auth, wait_on_rate_limit=True)

    try:
        print('Extracting Tweet')

        # Creation of query method using parameters. -filter removes retweets
        tweets = tweepy.Cursor(api.search_tweets, q=f'{tweet_text} -filter:retweets', tweet_mode='extended'
                               ).items(tweet_count)

        # Pulling information from tweets iterable object
        tweet_info = [[tweet.created_at, tweet.user.screen_name, remove_mention(tweet.full_text)] for tweet in tweets]

        # Creation of dataframe from tweets list
        tweets_df = pd.DataFrame(tweet_info, columns=['Created Date', 'Tweeted By', 'Text'])
        return tweets_df

    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)


def remove_mention(tweet_content):
    mention_removed = re.sub(r'@\w+', '', tweet_content)
    return mention_removed
