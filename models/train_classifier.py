"""This module contains the train_classifier.py script which is used to train the machine learning model.

   Usage:
      python train_classifier.py <database_filepath> <model_filepath>

   Example:
      python train_classifier.py ../data/DisasterResponse.db classifier.pkl
"""
import string
import sys

import joblib
import pandas as pd
from joblib import parallel_backend
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_training_data(database_filepath: str) -> (DataFrame, DataFrame, list[str]):
    """Loads the data from the given sqlite database file and returns the X (messages) and Y (categories)
       dataframes as well as the category names of Y as a list.

       :param database_filepath: The filepath to the SQLite database.
       :return: The X and Y dataframes as well as the classification category names of Y as a list.
    """

    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("messages", engine)
    X = df["message"]

    categories = list(df.columns)

    # remove all non category columns
    categories.remove("id")
    categories.remove("message")
    categories.remove("original")
    categories.remove("genre")

    Y = df[categories]

    return (X, Y, categories)


def tokenize(text: str) -> list[str]:
    """Tokenizes the given text by:
        1. Converting to lowercase
        2. Removing punctuation
        3. Word tokenizing the text
        4. Removing stop words
        5. Lemmatizing the tokens
       :param text: The text to tokenize.
       :return: The tokenized text as list of tokens.
    """

    # Normalize text by converting to lowercase and removing punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def build_model_pipeline(n_jobs : int = 1) -> Pipeline:
    """Builds the machine learning pipeline and returns it. The pipeline consists of:

        1. CountVectorizer with the tokenize function as tokenizer
        2. TfidfTransformer
        3. MultiOutputClassifier with a RandomForestClassifier as estimator

    :param n_jobs: The number of jobs to run in parallel. -1 means using all processors.
    :return: The machine learning pipeline.
    """

    # CAUTION: the n_jobs parameter requires the model fit to be wrapped in parallel_backend
    #          'multiprocessing' to work properly.
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=n_jobs)))
    ])

    return pipeline


def create_classification_report_per_category(Y_test: DataFrame, Y_pred: DataFrame, category_names: list[str]) -> str:
    """Creates a classification report for each category and returns the full report as a string.

    :param Y_test: The test data.
    :param Y_pred: The predicted data.
    :param category_names: The category names.
    :return: The classification report as a string.
    """

    result_string = ""
    for i, category in enumerate(category_names):
        result_string += f"Category: {category}\n"
        result_string += classification_report(Y_test.iloc[:, i], Y_pred[:, i]) + "\n"
    return result_string


def save_report_to_file(report: str, filepath: str):
    """Saves the report to the given file.

    :param report: The report to save.
    :param filepath: The filepath to save the report to.
    """
    with open(filepath, 'w') as file:
        file.write(report)


def evaluate_model(model, X_test: DataFrame, Y_test: DataFrame, category_names: list[str]) -> str:
    """Evaluates the given model on the given test data and prints and saves the classification report.

    :param model: The model to evaluate.
    :param X_test: The test data.
    :param Y_test: The test labels.
    :param category_names: The category names.
    """
    # Predict on the testing set
    Y_pred = model.predict(X_test)

    # Classification report for each category
    report = create_classification_report_per_category(Y_test, Y_pred, category_names)

    return report


def save_model_to_file(model: Pipeline, file_path: str):
    """Saves the model to the given file path using the joblib serializer.

    :param model: The model to save.
    :param file_path: The file path to save the model to.
    """
    joblib.dump(model, file_path)


def main():
    """Main method of the train_classifier.py script. Loads the data from the given database file, builds the model
       pipeline, trains the model, evaluates the model and saves the model to the given file path."""

    # Parse command line arguments
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f"Loading training data from sqlite database '{database_filepath}' ...")
        X, Y, classification_category_names = load_training_data(database_filepath)

        print("Splitting data into training and test set ...")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        USE_ALL_CPUS = -1

        # Using all CPUs for training, set number_of_parallel_jobs to 1 if you run into issues.
        number_of_parallel_jobs = USE_ALL_CPUS

        # CAUTION: A n_jobs parameter greater than 1 requires the model fit to be wrapped in
        #           parallel_backend 'multiprocessing' to work properly.
        with parallel_backend('multiprocessing'):

            print("Building model...")
            model_pipeline = build_model_pipeline(n_jobs=number_of_parallel_jobs)

            print("Training model...")
            # Define the parameter grid
            parameters = {
                'vect__max_df': (0.8, 0.9, 1.0),
                'clf__estimator__n_estimators': [50, 100, 200],
            }
            # Initialize GridSearchCV
            grid_search = GridSearchCV(model_pipeline, param_grid=parameters, n_jobs=USE_ALL_CPUS, verbose=2)

            # Fit and tune model
            optimized_model = grid_search.fit(X_train, Y_train)

        print('Evaluating model ...')
        classification_report = evaluate_model(optimized_model, X_test, Y_test, classification_category_names)
        print(classification_report)

        classification_report_file = "classification_report.txt"
        print(f"--> Saving classification report to file '{classification_report_file}' ...")
        save_report_to_file(classification_report, classification_report_file)

        print(f"Saving model ... {model_filepath}")
        save_model_to_file(optimized_model, model_filepath)

        print("TRAINING COMPLETED!")
        print(f"Model saved to file '{model_filepath}'")

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
