from flask import Flask, request, Response
import datetime
import json
import csv
import os
import pickle
import psycopg2.extras as extras
import psycopg2
from load_test_dataset import load_batch_data
from flask_cors import CORS, cross_origin
import pandas as pd
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)
cors = CORS(app, resources={
     r"/*" : {
       "origins":"*"
     }

})
ip = None
with open('docker_vm_ip.json') as f:
   ip = json.load(f)['docker_vm_ip']
def age_preprocess(df):

        df['Age'] = df['Age'].apply(lambda x : float(x))
        df['Age'][df['Age'] <= 16] = 0
        df['Age'][(df['Age'] > 16) & (df['Age'] <= 26)] = 1
        df['Age'][(df['Age'] > 26) & (df['Age'] <= 36)] = 2
        df['Age'][(df['Age'] > 36) & (df['Age'] <= 62)] = 3
        df['Age'][ df['Age'] > 62] = 4
        return df

def gender_preprocess(df):
    sex_mapping = {"male": 0, "female": 1}
    df['Sex'] = df['Sex'].map(sex_mapping)
    return df

def fare_preprocess(df):

    df['Fare'] = df['Fare'][df['Fare'] != None]
    df['Fare'] = df['Fare'].apply(lambda x : float(x))
    df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    df['Fare'][ df['Fare'] <= 17] = 0
    df['Fare'][(df['Fare'] > 17) & (df['Fare'] <= 30)] = 1
    df['Fare'][(df['Fare'] > 30) & (df['Fare'] <= 100)] = 2
    df['Fare'][ df['Fare'] > 100] = 3
    return df

def cabin_preprocessing(df):
    df = df[~df['Cabin'].isna()]
    df = df[df['Cabin'] != 'nan']
    df = df[df['Cabin'] != None]

    df['Cabin'] = df['Cabin'].str[:1]

    cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

    df['Cabin'] = df['Cabin'].map(cabin_mapping)
    return df



def embark_preprocess(df):

    df['Embarked'] = df['Embarked'].fillna('S')

    #####51
    embarked_mapping = {"S": 0, "C": 1, "Q": 2}

    df['Embarked'] = df['Embarked'].map(embarked_mapping)
    return df

def title_preprocess(df):

    df['Name'] = df['Name'].apply(lambda x :str(x))
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    ###24
    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                     "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                     "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

    df['Title'] = df['Title'].map(title_mapping)
    return df

def familySize_preprocessing(df):

    df['SibSp'] = df['SibSp'].apply(lambda x : float(x))
    df['Parch'] = df['Parch'].apply(lambda x : float(x))
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

    df['FamilySize'] = df['FamilySize'].map(family_mapping)
    return df


def getPostgresConnection_load():
    conn = psycopg2.connect(
        host=ip,
        database="docker",
        user="docker",
        password="docker",
        port=5432)
    return conn

def getPostgresConnection_load_target():
    conn = psycopg2.connect(
        host=ip,
        database="docker",
        user="docker",
        password="docker",
        port=5434)
    return conn




def load_data_into_target_db(df):
    df2 = df
    for x in df.columns:
        df2[x] = df2[x].apply(lambda x : str(x))
    print (df2.dtypes)
    print (df2)
    conn = getPostgresConnection_load_target()
    cur = conn.cursor()
    cur.execute('CREATE SCHEMA if not exists eder;')
    create_table = """create table if not exists eder.result_dataset(
                        PassengerId  varchar(5),
                        Pclass varchar(1),
                        Name varchar(200),
                        Sex varchar(10),
                        Age varchar(10),
                        SibSp varchar(1),
                        Parch varchar(1),
                        Ticket varchar(100),
                        Fare varchar(20),
                        Cabin varchar(100),
                        Embarked varchar(5),
                        Prediction varchar(5),
                        Process_time varchar(100)
                        );"""
    cur.execute(create_table)
    tuples = [tuple(x) for x in df2.to_numpy()]
    print (tuples)
    cols = ','.join(list(df2.columns))
    query  = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s)" % ('eder.result_dataset', cols)
    cur.executemany(query, tuples)

    conn.commit()
    cur.close()
    conn.close()


def load_predict(data):
    Pclass = data['Pclass']
    Sex = data['Sex']
    Age = data['Age']
    Fare = data['Fare']
    Cabin = data['Cabin']
    Embarked = data['Embarked']
    Title = data['Title']
    FamilySize = data['FamilySize']
    Model = data['Model'].strip()
    print (data)
    model = None
    if Model.upper() == 'SVC':
        model = pickle.load(open('model/SVC_model.sav', 'rb'))
    elif Model.upper() == 'DTC':
        model = pickle.load(open('model/DTC_model.sav', 'rb'))
    elif Model.upper() == 'GNB':
        model = pickle.load(open('model/GNB_model.sav', 'rb'))
    elif Model.upper() == 'KNN':
        model = pickle.load(open('model/KNN_model.sav', 'rb'))
    elif Model.upper() == 'RFC':
        model = pickle.load(open('model/RFC_model.sav', 'rb'))
    else :
        msg_dict = {}
        msg_dict['msg'] = 'No such model exists'
        return Response(json.dumps(msg_dict, ensure_ascii=False), mimetype='text/json', status=200)
    data.pop('Model', None)
    data_arr = [[ Pclass   , Sex , Age  , Fare   , Cabin  ,    Embarked ,  Title  ,   FamilySize]]
    test_df = pd.DataFrame(data_arr,columns = ['Pclass' , 'Sex' , 'Age' , 'Fare' , 'Cabin' , 'Embarked' , 'Title' , 'FamilySize'])
    pred = model.predict(test_df)
    msg_dict = {}
    msg_dict['msg'] = 'prediction sucessful'
    msg_dict['attributes'] = data
    msg_dict['model'] = Model.upper()
    msg_dict['prediction'] = int(pred[0])
    return Response(json.dumps(msg_dict, ensure_ascii=False), mimetype='text/json', status=200)


def batch_pridict_load(df,Model):
    model = None
    if Model.upper() == 'SVC':
        model = pickle.load(open('../model/SVC_model.sav', 'rb'))
    elif Model.upper() == 'DTC':
        model = pickle.load(open('../model/DTC_model.sav', 'rb'))
    elif Model.upper() == 'GNB':
        model = pickle.load(open('../model/GNB_model.sav', 'rb'))
    elif Model.upper() == 'KNN':
        model = pickle.load(open('../model/KNN_model.sav', 'rb'))
    elif Model.upper() == 'RFC':
        model = pickle.load(open('../model/RFC_model.sav', 'rb'))
    else :
        msg_dict = {}
        msg_dict['msg'] = 'No such model exists'
        return Response(json.dumps(msg_dict, ensure_ascii=False), mimetype='text/json', status=200)

    pred = model.predict(df)
    return pred





def batch_pridict(psg_start,psg_end,model):
    conn = getPostgresConnection_load()
    cur = conn.cursor()
    cur.execute('select * from eder.test_dataset where cast(PassengerId AS numeric) >= {startIndex} and cast(PassengerId AS numeric) <= {endIndex};'.format(startIndex = int(psg_start),endIndex=int(psg_end)))
    data = cur.fetchall()
    df = pd.DataFrame(data,columns = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
    for col in df.columns:
        df[col] = df[col][df[col] != 'nan']
    df_copy = df
    df = age_preprocess(df)
    df = gender_preprocess(df)
    df = fare_preprocess(df)
    df = cabin_preprocessing(df)
    df =  embark_preprocess(df)
    df = title_preprocess(df)
    df =  familySize_preprocessing(df)
    cols_need = ['Pclass' , 'Sex' , 'Age' , 'Fare' , 'Cabin' , 'Embarked' , 'Title' , 'FamilySize']
    df = df.drop([x for x in df.columns if x not in cols_need],axis =1)
    if len(df) > 0:
       pred = batch_pridict_load(df,model)
       PRED = []
       for x in pred:
           PRED.append([x])
       df_pred = pd.DataFrame(PRED,columns = ['Prediction'])
       df_copy['Prediction'] = df_pred['Prediction']
       df_copy['process_time'] = str(datetime.datetime.now())
       load_data_into_target_db(df_copy)


@app.route('/load_dataset_to_db',methods = ['GET'])
def load_initial():
    if request.method == 'GET':
        load_batch_data()
        return "done"


@app.route('/predict_batch',methods = ['GET'])
def batch():
    if request.method == 'GET':
        data  = request.get_json()
        batch_pridict(data['psg_start'],data['psg_end'],data['model'])
        return "done"


@app.route('/predict_single',methods = ['GET'])
def single():
    if request.method == 'GET':
        data  = request.get_json()
        response = load_predict(data)
        return response


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
