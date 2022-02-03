import pandas as pd
import pickle
import constants
import pre_processor


# Add function to extract live tweets based on topic later and feed to predict data

def predict_data(model_name, tweet_to_predict=constants.TEST_TWEET_TO_PREDICT):
    if model_name not in constants.VALID_MODELS:
        raise Exception(f'Invalid model. Model has to one of {constants.VALID_MODELS}')
    vectorizer = pickle.load(open('trained_models/trained_vectorizer_model.pk', 'rb'))
    model = pickle.load(open(f'trained_models/trained_{model_name}_model.pk', 'rb'))

    tweet_df = pd.DataFrame({'Sentiment': [0]*len(tweet_to_predict), 'Text': tweet_to_predict})
    processed_tweet = pre_processor.pre_process_data(tweet_df)
    vectorized_tweet = vectorizer.transform(processed_tweet.Text)
    prediction = model.predict(vectorized_tweet)
    prediction_value = [constants.SENTIMENT_DICT[key] for key in prediction]
    print('-'*102)
    prediction_df = pd.DataFrame({'Tweet': tweet_to_predict, 'Sentiment': prediction_value})
    print(prediction_df)
