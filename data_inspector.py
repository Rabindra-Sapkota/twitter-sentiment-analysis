import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(file_name, header=None, encoding='ISO-8859-1'):
    """
    Loads file with pre-defined Headers and Encoding

    Args:
        file_name (str): Name of the file that is to be loaded
        header (list): Header of the dataframe
        encoding (str): Encoding of data

    Returns:
        dataframe: Dataframe from the given file name.
    """
    try:
        data_frame = pd.read_csv(file_name, names=header, encoding=encoding)
        return data_frame
    except FileNotFoundError:
        exception_message = f'File {file_name} is not found.'
        print(exception_message)
        raise Exception(exception_message)
    except Exception as e:
        print(e)
        raise Exception('Unable to read data')


def eda_of_dataframe(df):
    """
    Takes dataframe and describes its attributes

    Args:
         df (dataframe): Dataframe whose attribute has to be described

    Returns:
        None
    """
    # pd.set_option('display.max_rows', None, 'display.max_columns', None)
    # pd.set_option('mode.chained_assignment', None)
    print('-'*50, 'Sample five sample data', '-'*50)
    print(df.sample(5))
    print('-'*50, 'Number of Unique values for data', '-'*50)
    print(df.nunique())
    print('-'*50, 'Data Size', '-'*50)
    data_size = df.shape
    print(f'Rows: {data_size[0]}, Columns: {data_size[1]}')
    print('-'*50, 'Null Values Count', '-'*50)
    print(df.isna().sum())
    print('-'*120)


def visualize_target_distribution(df_series):
    """
    Visualize portion of data for negative and positive tweets for given series
    Args:
        df_series(data_series): Data Series whose distribution is to be visualized
    """
    data_distribution = df_series.groupby(df_series).count()
    ax = data_distribution.plot(kind='bar', title='Distribution of Sentiment in data')
    ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
    plt.show()
