# Disaster Response Pipeline Project

### Introduction:
This project is for categorizing messages for Disaster responses based on a machine learning model. This is useful in aligning the right resource to the responses in an effective and  cost saving manner. 

### File structure

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app and loads the necessary graphs
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # File that loads the dat from csvs and inserts into the Database
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py #Loads the data from database and trains a ML model on it. 
|- classifier.pkl # saved model
README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
