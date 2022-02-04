from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer, WordNetLemmatizer
import re
from data_inspector import pd


def pre_process_data(df):
    """
    Pre-process data. Does encoding, case-equalization
    Args:
        df (dataframe): DataFame which is to be pre-processed
    Returns:
        dataframe: pre-processed data frame
    """
    pd.set_option('mode.chained_assignment', None)
    # Take only samples
    # df = df.sample(10000)

    # Encode Sentiment value as 1 and 0
    df['Sentiment'] = df.Sentiment.replace(4, 1)

    # Stem, lemmatize, tokenize data with cleansing. case conversion done implicitly
    df['Text'] = df.Text.apply(lambda text: stem_lemmatize_tokenize_with_cleaning(text))
    return df


def stem_lemmatize_tokenize_with_cleaning(string_with_stop_word):
    """
    Remove Stop words, URL and punctuation from the string. Returns Tokenized data

    Args:
         string_with_stop_word(str): String with stop word characters

    Returns:
        set: Set with string element after removal of stop words
    """
    st = PorterStemmer()
    lm = WordNetLemmatizer()
    eng_stop_word = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')

    string_after_url_remover = re.sub(r'(www.[\S]*)|(https?://[\S]*)', ' ', string_with_stop_word)
    string_list = tokenizer.tokenize(string_after_url_remover)
    cleaned_list = [lm.lemmatize(st.stem(string)) for string in string_list if string not in eng_stop_word]
    return ' '.join(cleaned_list)
