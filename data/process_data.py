"""This script loads disaster messages and the manually classified categories from CSV files and merges them
into one dataframe. The data is then cleaned and saved to a SQLite database.

process_data.py [DISASTER_MESSAGE_CSV] [MESSAGE_CATEGORIES_CSV] [SQLITE_DB_FILE]

Usage:
   python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the data from the given filepaths and merges them into one dataframe.

       :param messages_filepath: The filepath to the CSV file containing the messages.
       :param categories_filepath: The filepath to the CSV file containing the categories.
       :return: A dataframe containing the merged data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Split the 'categories' column by semicolon (;)
    categories['categories'] = categories['categories'].str.split(';')

    # Extract the column headers from the first row by splitting away the last two characters
    # and removing the duplicates
    headers = categories['categories'].iloc[0]
    headers_clean = set(["".join(header[:-2]) for header in headers])

    for category in headers_clean:
        categories[category] = categories['categories'].apply(lambda x: int(category + '-1' in x))

    # Drop the original 'categories' column and only keep the expanded categories
    categories.drop('categories', axis=1, inplace=True)

    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """Cleans the data by removing duplicates.

    :param df: The dataframe to clean.
    :return: The cleaned dataframe.
    """
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Saves the data to a SQLite database.
    :param df: The dataframe to save.
    :param database_filename: The filename of the SQLite database.
    :return: The dataframe that was saved.
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('messages', engine, index=False)
    return df


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
