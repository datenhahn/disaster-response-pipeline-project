# Project: Disaster Response Pipeline

![Flooded houses as image for disaster](.assets/disaster.png)

The goal of this project is to build a Natural Language Processing (NLP) model to categorize messages sent during disasters. The model is trained on a dataset provided by Appen (formally Figure 8) containing real messages that were sent during disaster events. The model is then used to classify new messages.

## Requirements

Tested with the following versions, but might also work with other versions.

* Python 3.10.6

Libraries:

* pandas==2.0.1
* sqlalchemy==2.0.15

## Run Project

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

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
