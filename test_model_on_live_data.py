import pandas as pd
import pickle
import constants
import pre_processor
import extract_twitter_data
import time


def predict_data(model_name, tweet_keyword=None, tweet_count=constants.TWEET_EXTRACTION_COUNT, to_csv=False):
    if tweet_keyword is None:
        sample_df = get_non_twitter_sample_data()
    else:
        try:
            sample_df = extract_twitter_data.extract_tweets(tweet_keyword, tweet_count)
            sample_df['Sentiment'] = 0
        except Exception as e:
            print(e)
            print('Failed to get twitter data. Proceeding with normal')
            sample_df = get_non_twitter_sample_data()

    if model_name not in constants.VALID_MODELS:
        raise Exception(f'Invalid model. Model has to one of {constants.VALID_MODELS}')

    vectorizer = pickle.load(open('trained_models/trained_vectorizer_model.pk', 'rb'))
    model = pickle.load(open(f'trained_models/trained_{model_name}_model.pk', 'rb'))

    processed_tweet = pre_processor.pre_process_data(sample_df.copy())
    vectorized_tweet = vectorizer.transform(processed_tweet.Text)
    prediction = model.predict(vectorized_tweet)
    prediction_value = [constants.SENTIMENT_DICT[key] for key in prediction]
    print('-'*102)
    sample_df['Sentiment'] = prediction_value
    print(sample_df)
    print('-'*102)
    if to_csv:
        file_name = time.strftime("dataset/ModelPredictionOutput%Y%m%d_%H%M%S.csv")
        print(file_name)
        sample_df.to_csv(file_name, index=False)


def get_non_twitter_sample_data():
    sample_data = constants.TEST_TWEET_TO_PREDICT
    sample_df = pd.DataFrame({'Sentiment': [0] * len(sample_data), 'Text': sample_data})
    return sample_df
