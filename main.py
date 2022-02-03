import pickle
import constants
import data_inspector
import pre_processor
# import post_visualize
import train_model
import test_model_on_live_data


def train_data_model():
    """
    Train the data with specified model. Save Trained Model to file and print test stat
    """
    print('-'*102)
    print('Reading data from file')
    twitter_df = data_inspector.load_dataset(constants.TWITTER_DATASET, constants.TWITTER_DATA_HEADER)
    twitter_df = twitter_df.drop(columns=['ids', 'date', 'flag', 'user'])

    print('Inspecting data')
    data_inspector.eda_of_dataframe(twitter_df)
    data_inspector.visualize_target_distribution(twitter_df.Sentiment)

    print('Pre-Processing Data')
    twitter_preprocessed = pre_processor.pre_process_data(twitter_df)

    print('Word plot for most occurred positive and negative words')
    # post_visualize.negative_word_plot(twitter_preprocessed)
    # post_visualize.positive_word_plot(twitter_preprocessed)

    # Valid model names are: svc, bernoulli, logistic_regression
    print('Training Model')
    trained_model = train_model.train_and_evaluate_model(twitter_preprocessed, constants.MODEL_TO_TRAIN)
    pickle.dump(trained_model, open(f'trained_models/trained_{constants.MODEL_TO_TRAIN}_model.pk', 'wb'))
    print('-'*102)


def test_model():
    """
    Run prediction with pre-trained model
    """
    test_model_on_live_data.test_model(constants.MODEL_TO_TRAIN)


if __name__ == '__main__':
    print(f'Running Program in {constants.RUN_METHOD} mode')
    if constants.RUN_METHOD == 'Train':
        train_data_model()
    elif constants.RUN_METHOD == 'Predict':
        test_model()
    else:
        raise Exception('Invalid run method. Run method is either train or predict')
