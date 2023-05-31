import json
import os

import joblib
import pandas as pd
import plotly

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

# this import is required for deserialization of the model
from models.train_classifier import tokenize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

db_file = f'sqlite://{BASE_DIR}/data/DisasterResponse.db'
# load data
print(f"Loading database: {db_file}")
engine = create_engine(db_file)
df = pd.read_sql_table('messages', engine)

# load model

# Note: the tokenize function is required for deserialization of the model, we execute a dummy
# statement to make sure the import is not removed by code tools.
tokenize("test")
model = joblib.load(f"{BASE_DIR}/models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    """Renders the index page of the web app. The page contains two visualizations of the data and provides a form
       to enter a message to classify.
    """

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """Web page that handles user query and displays model results"""
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
