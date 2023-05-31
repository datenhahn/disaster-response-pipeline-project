# Project: Disaster Response Pipeline

![Flooded houses as image for disaster](.assets/disaster.png)

The goal of this project is to build a Natural Language Processing (NLP) model to categorize messages sent during disasters. The model is trained on a dataset provided by Appen (formally Figure 8) containing real messages that were sent during disaster events. The model is then used to classify new messages.

The project consists of three parts:

* ETL Pipeline: Loads the messages and categories datasets, merges the two datasets, cleans the data, and stores it in a SQLite database.
* ML Pipeline: Loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline and exports the final model as a pickle file.
* Flask Web App: Loads the SQLite database and the trained model and provides a web interface to classify new messages.

**Quickstart (Running the Web-App)**

To run the project with the pretrained model and prepared database execute:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app/run.py
```

Go to http://0.0.0.0:3001/ to access the webapp.

**Training the Model**

The Quickstart section above uses a pretrained model and a prepared database.

If you want to train the model yourself, the following steps are required, which are explained in more detail below.

* Setup a virtual environment and install the required libraries.
* Prepare the message database (ETL Pipeline).
* Train the model (ML Pipeline).


## Technical Requirements

Tested with the following versions, but might also work with other versions.

* Python 3.10.6

Libraries:

* pandas==2.0.1
* sqlalchemy==2.0.15
* jupyter==1.0.0
* nltk==3.8.1
* scikit-learn==1.2.2
* plotly==5.14.1
* flask==2.3.2
* pytest==7.3.1

## About the Dataset

The dataset contains real messages that were sent during disaster events. The messages were collected by Appen (formally Figure Eight) and were pre-labelled by them into 36 categories. The dataset contains the original message as well as the translated message.

The dataset contains 26,248 messages in total from different sources.

* Direct Communication: 10766
* News: 13054
* Social Media: 2396

The following categories are available:

Floods, Hospitals, Aid Related, Child Alone, Related, Water, Food, Weather Related, Fire, Shops, Direct Report, Military, Aid Centers, Other Aid, Cold, Shelter, Storm, Other Weather, Offer, Money, Missing People, Tools, Search And Rescue, Earthquake, Death, Buildings, Other Infrastructure, Infrastructure Related, Refugees, Request, Medical Products, Security, Clothing, Medical Help, Transport, Electricity



## Training the Model

### Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Prepare Message Database

Merge the disaster messages and the manually classified categories into one database.

After this step the file `data/DisasterResponse.db` should have been created.

```bash
cd data
python ./process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

### Train the Model

This section describes the training of a model with the parameters used in the web-app.

As well as the exploration of different parameter combinations
to find a better model.

_Make sure you have the virtual environment prepared and activated before._

#### A) Train the model with the training pipeline

The train_classifier.py script trains the model with data
from the sqlite database prepared in the previous step.

It takes two arguments: The path to the sqlit database and the path to the pickle file to store the model.

```bash
cd models
python ./train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

#### B) Explore different parameter combinations or try out new models

The `models` folder also contains a jupyter notebook `ML Pipeline Preparation.ipynb` that can be used to explore different parameter combinations or try out new models.

During development the following parameters were tested with
the notebook in a grid search:

```
parameters = {
    'vect__max_df': (0.8, 0.9, 1.0),
    'clf__estimator__n_estimators': [50, 100, 200],
}

with parallel_backend('multiprocessing'):
    # Initialize GridSearchCV
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=cpus, verbose=2)

    # Fit and tune model
    cv.fit(X_train, Y_train)
```

It turned out that the default parameters of the CountVectorizer and the RandomForestClassifier were the best.

```
cv.best_params_
{'clf__estimator__n_estimators': 100, 'vect__max_df': 1.0}
```

Feel free to experiment with other vectorizers or classifiers.


## License

Copyright 2023 Ecodia GmbH & Co. KG

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.