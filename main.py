import constants
import pre_processor


def main():
    # pd.set_option('display.max_colwidth', None)
    twitter_df = pre_processor.load_dataset(constants.TWITTER_DATASET, constants.TWITTER_DATA_HEADER)
    twitter_df = twitter_df.drop(columns=['ids', 'date', 'flag', 'user'])
    pre_processor.eda_of_dataframe(twitter_df)
    pre_processor.visualize_target_distribution(twitter_df.Sentiment)


if __name__ == '__main__':
    main()
