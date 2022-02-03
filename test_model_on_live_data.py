import pandas as pd
import pickle
import constants
import pre_processor


# add function to extract live tweets later
live_tweets = ['I am very fucking happy']*10


def test_model(model_name):
    if model_name not in constants.VALID_MODELS:
        raise Exception(f'Invalid model. Model has to one of {constants.VALID_MODELS}')
    vectorizer = pickle.load(open('trained_models/trained_vectorizer_model.pk', 'rb'))
    model = pickle.load(open(f'trained_models/trained_{model_name}_model.pk', 'rb'))

    tweet_df = pd.DataFrame({'Sentiment': [0]*10, 'Text': live_tweets})
    processed_tweet = pre_processor.pre_process_data(tweet_df)
    vectorized_tweet = vectorizer.transform(tweet_df)
    print(tweet_df.shape)
    print(vectorized_tweet.shape)
    prediction = model.predict(vectorized_tweet)
    # print(dir(prediction))
    # print(f'Tweet {tweet} has {prediction} sentiment')
    print(prediction)
