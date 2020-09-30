import psycopg2
import psycopg2.extras as extras
import pandas as pd
import json


def getPostgresConnection_load():
    ip = None
    with open('docker_vm_ip.json') as f:
       ip = json.load(f)['docker_vm_ip']
    conn = psycopg2.connect(
        host=ip,
        database="docker",
        user="docker",
        password="docker",
        port=5432)
    return conn
def load_batch_data():
    df = pd.read_csv('test.csv')
    for x in df.columns:
        df[x] = df[x].apply(lambda x : str(x))

    print (type(df))
    conn = getPostgresConnection_load()
    cur = conn.cursor()
    cur.execute('CREATE SCHEMA if not exists eder;')
    cur.execute('drop table if exists eder.test_dataset;')

    create_table = """create table eder.test_dataset(
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
                    Embarked varchar(5)
                    );"""
    cur.execute(create_table)
    print ('Flag 1')
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ','.join(list(df.columns))
    print (cols)
    query  = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s,%%s)" % ('eder.test_dataset', cols)
    cur.executemany(query, tuples)

    conn.commit()
    cur.close()
    conn.close()




#load_batch_data()
