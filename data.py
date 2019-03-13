from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from misc import *
import numpy as np
import pandas as pd

es = Elasticsearch(hosts="10.8.0.3")

def get_training_data(spark):
    # get es data

    s = Search(using=es, index="*").filter("term", type="flow").filter('range', **{'@timestamp': {'gte': 'now-5m' , 'lt': 'now'}})

    response = s.scan()
    # load training features
    train_features = []
    for hit in response:
        if 'port' in hit.source and 'ip' in hit.source:
            #print(ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port)
            train_features.append([ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port])

    # create labels for training data
    train_labels = []
    for feature in train_features:
        if feature[1] == 9200 or feature[1] == 5601 or feature[3] == 9200 or feature[3] == 5601:
            train_labels.append(0)
        else:
            train_labels.append(1)

    # set columns
    columns_ip = "source_ip source_port dest_ip dest_port".split(' ')

    # create feature dataframe
    train_features = np.asarray(train_features)
    train_features_df = pd.DataFrame(train_features, columns=columns_ip)
    train_features_df = spark.createDataFrame(train_features_df)

    # create label dataframe
    train_labels = np.asarray(train_labels)
    train_labels_df = pd.DataFrame(train_labels, columns=['label'])
    train_labels_df = spark.createDataFrame(train_labels_df)

    # create feature vectors column
    vecAssembler = VectorAssembler(inputCols=["source_ip", "source_port", "dest_ip", "dest_port"], outputCol="features")
    train_features_df = vecAssembler.transform(train_features_df)

    #train_features_df.show()
    #train_labels_df.show()
    return train_features, train_labels, train_features_df, train_labels_df

def get_live_data(spark):
    # pull live data set from es
    es = Elasticsearch(hosts="10.8.0.3")
    s = Search(using=es, index="*").filter("term", type="flow").filter('range', **{'@timestamp': {'gte': 'now-1h' , 'lt': 'now'}})

    #s.aggs.bucket('by_timestamp', 'terms', field='@timestamp', size=999999999).metric('total_net_bytes', 'sum', field="source.stats.net_bytes_total")

    response = s.scan()

    data = []
    for hit in response:
        if 'port' in hit.source and 'ip' in hit.source:
            #print(ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port)
            data.append([ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port])

    # create dataframes for live data
    data = np.asarray(data)
    columns = ["source_ip", "source_port", "dest_ip", "dest_port"]
    data_pddf = pd.DataFrame(data, columns=columns)
    data_df = spark.createDataFrame(data_pddf)

    vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
    data_df = vecAssembler.transform(data_df)
    return(data_df)

def get_predicted_data(spark):
    # pull live data set from es
    es = Elasticsearch(hosts="10.8.0.3")
    s = Search(using=es, index="model_predictions*").filter('range', **{'@timestamp': {'gte': 'now-5m' , 'lt': 'now'}})

    #s.aggs.bucket('by_timestamp', 'terms', field='@timestamp', size=999999999).metric('total_net_bytes', 'sum', field="source.stats.net_bytes_total")

    #['@timestamp', '@version', 'data', 'dest_ip', 'dest_port', 'features', 'indexedFeatures', 'meta', 'predictedLabel', 'prediction', 'probability', 'rawPrediction', 'source_ip', 'source_port', 'tags']

    response = s.scan()

    data = []
    for hit in response:
        if "predictedLabel" in hit:
            data.append([int(hit.source_ip), int(hit.source_port), int(hit.dest_ip), int(hit.dest_port), int(hit.predictedLabel)])

    columns = ["source_ip", "source_port", "dest_ip", "dest_port", "label"]
    data = np.asarray(data)
    data_pddf = pd.DataFrame(data, columns=columns)
    data_df = spark.createDataFrame(data_pddf)
    vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
    data_df = vecAssembler.transform(data_df)
    return(data_df)

