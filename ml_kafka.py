#!/usr/bin/env python
# coding: utf-8

import os
import socket
import struct

import numpy as np
import pandas as pd
import pyspark
from apscheduler.schedulers.background import BackgroundScheduler
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from pyod.utils.data import get_outliers_inliers
from pyspark import SQLContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.types import *

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    '--conf "spark.redis.host=10.8.0.7" --jars /home/user/spark-redis-2.3.1-SNAPSHOT-jar-with-dependencies.jar --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.0,org.elasticsearch:elasticsearch-hadoop:6.6.1 pyspark-shell'
# '--master spark://10.8.0.9:7077 --total-executor-cores 4 --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.0,org.elasticsearch:elasticsearch-hadoop:6.6.1 pyspark-shell'

sc = pyspark.SparkContext(appName="ml_kafka")
sc.setLogLevel("ERROR")
spark = SQLContext(sc)

es = Elasticsearch(hosts="10.8.0.3")


def train_dt_model():  # decision tree with pipeline

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(ml_df)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(ml_df)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = ml_df.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, dt])

    # Train model.  This also runs the indexers.
    dt_model = pipeline.fit(trainingData)
    return dt_model


def get_training_data():
    # get es data

    s = Search(using=es, index="*").filter("term", type="flow").filter('range',
                                                                       **{'@timestamp': {'gte': 'now-5m', 'lt': 'now'}})

    response = s.scan()
    # load training features
    train_features = []
    for hit in response:
        if 'port' in hit.source and 'ip' in hit.source:
            # print(ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port)
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

    # train_features_df.show()
    # train_labels_df.show()
    return train_features, train_labels, train_features_df, train_labels_df


def get_outliers(train_features, train_labels):
    # outlier detection
    x_outliers, x_inliers = get_outliers_inliers(train_features, train_labels)
    for i in x_outliers[0:10]:
        print("src_ip: {srcip}, src_port: {srcport}, dst_ip: {dstip}, dst_port: {dstport}".format(srcip=long2ip(i[0]),
                                                                                                  srcport=i[1],
                                                                                                  dstip=long2ip(i[2]),
                                                                                                  dstport=i[3]))


# kmeans clustering unsupervised
def get_kmeans_model(train_features_df):
    kmeans = KMeans(k=10, seed=1)
    kmeans_model = kmeans.fit(train_features_df.select('features'))

    return kmeans_model


# random forest classifier
def get_rf_model(train_features_df):
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(train_features_df)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(
        train_features_df)

    # Split the data into training and test sets (30% held out for testing)
    (training_data, test_data) = train_features_df.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    label_converter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                    labels=label_indexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, rf, label_converter])

    # Train model.  This also runs the indexers.
    rf_model = pipeline.fit(training_data)
    return rf_model


def get_live_data():
    # pull live data set from es
    es = Elasticsearch(hosts="10.8.0.3")
    s = Search(using=es, index="*").filter("term", type="flow").filter('range',
                                                                       **{'@timestamp': {'gte': 'now-1h', 'lt': 'now'}})

    # s.aggs.bucket('by_timestamp', 'terms', field='@timestamp', size=999999999).metric('total_net_bytes', 'sum', field="source.stats.net_bytes_total")

    response = s.scan()

    data = []
    for hit in response:
        if 'port' in hit.source and 'ip' in hit.source:
            # print(ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port)
            data.append([ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port])

    # create dataframes for live data
    data = np.asarray(data)
    columns = ["source_ip", "source_port", "dest_ip", "dest_port"]
    data_pddf = pd.DataFrame(data, columns=columns)
    data_df = spark.createDataFrame(data_pddf)

    vec_assembler = VectorAssembler(inputCols=columns, outputCol="features")
    data_df = vec_assembler.transform(data_df)
    return data_df


def get_predicted_data():
    # pull live data set from es
    es = Elasticsearch(hosts="10.8.0.3")
    s = Search(using=es, index="model_predictions*").filter('range', **{'@timestamp': {'gte': 'now-5m', 'lt': 'now'}})

    # s.aggs.bucket('by_timestamp', 'terms', field='@timestamp', size=999999999).metric('total_net_bytes', 'sum', field="source.stats.net_bytes_total")

    # ['@timestamp', '@version', 'data', 'dest_ip', 'dest_port', 'features', 'indexedFeatures', 'meta', 'predictedLabel', 'prediction', 'probability', 'rawPrediction', 'source_ip', 'source_port', 'tags']

    response = s.scan()

    data = []
    for hit in response:
        if "predictedLabel" in hit:
            data.append([int(hit.source_ip), int(hit.source_port), int(hit.dest_ip), int(hit.dest_port),
                         int(hit.predictedLabel)])

    columns = ["source_ip", "source_port", "dest_ip", "dest_port", "label"]
    data = np.asarray(data)
    data_pddf = pd.DataFrame(data, columns=columns)
    data_df = spark.createDataFrame(data_pddf)
    vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
    data_df = vecAssembler.transform(data_df)
    return (data_df)


def get_user_predictions():
    # pull live data set from es
    es = Elasticsearch(hosts="10.8.0.3")
    s = Search(using=es, index="user_predictions*").filter('range', **{'@timestamp': {'gte': 'now-5m', 'lt': 'now'}})

    # s.aggs.bucket('by_timestamp', 'terms', field='@timestamp', size=999999999).metric('total_net_bytes', 'sum', field="source.stats.net_bytes_total")

    # ['@timestamp', '@version', 'data', 'dest_ip', 'dest_port', 'features', 'indexedFeatures', 'meta', 'predictedLabel', 'prediction', 'probability', 'rawPrediction', 'source_ip', 'source_port', 'tags']

    response = s.scan()

    data = []
    for hit in response:
        if "predictedLabel" in hit:
            data.append([int(hit.source_ip), int(hit.source_port), int(hit.dest_ip), int(hit.dest_port),
                         int(hit.predictedLabel)])

    columns = ["source_ip", "source_port", "dest_ip", "dest_port", "label"]
    data = np.asarray(data)
    data_pddf = pd.DataFrame(data, columns=columns)
    data_df = spark.createDataFrame(data_pddf)
    vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
    data_df = vecAssembler.transform(data_df)
    return (data_df)


def ip2long(ip):
    """
    Convert an IP string to long
    """
    if ip:
        packedIP = socket.inet_aton(ip)
        return struct.unpack("!L", packedIP)[0]


ip2long_udf = F.udf(ip2long, LongType())


def normalize_ips(df):
    df = df.withColumn('source_ip', F.col('data.json.src_ip')).withColumn('dest_ip',
                                                                          F.col('data.json.dest_ip')).withColumn(
        'source_ip', F.col('data.client_ip')).withColumn('dest_ip', F.col('data.ip')).withColumn('dest_ip', F.col(
        'data.dest.ip')).withColumn('source_ip', F.col('data.source.ip')).withColumn('dest_port', F.col(
        'data.dest.port')).withColumn('source_port', F.col('data.source.port')).filter('source_ip is not NULL').filter(
        'dest_ip is not NULL').filter('source_port is not NULL').filter('dest_port is not NULL')
    df = df.withColumn('source_ip', ip2long_udf(F.col('source_ip'))).withColumn('dest_ip',
                                                                                ip2long_udf(F.col('dest_ip')))
    return (df)


def process_batch(df, epoch_id):
    print(epoch_id)
    if epoch_id % 100 == 0:
        try:
            print('loading models')
            KMeansModel.load('hdfs://10.8.0.11:9000/data/models/kmeans_model')
            PipelineModel.load('hdfs://10.8.0.11:9000/data/models/rf_model')
        except Exception as e:
            print('unable to load modules, {}'.format(e))
    df = normalize_ips(df)
    # df.show()
    columns = ["source_ip", "source_port", "dest_ip", "dest_port"]
    vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
    df = vecAssembler.transform(df)
    # print(df)
    kmeans_model_result = kmeans_model.transform(df).withColumn('algo', F.lit('kmeans'))
    # kmeans_model_result.show()
    ds = kmeans_model_result.selectExpr("CAST('key' AS STRING)", "to_json(struct(*)) AS value").write.format(
        "kafka").option("kafka.bootstrap.servers", "10.8.0.8:9092").option("topic", "model_predictions").save()

    rf_model_result = rf_model.transform(df).withColumn('algo', F.lit('rf'))
    # rf_model_result.show()
    ds = rf_model_result.selectExpr("CAST('key' AS STRING)", "to_json(struct(*)) AS value").write.format(
        "kafka").option("kafka.bootstrap.servers", "10.8.0.8:9092").option("topic", "model_predictions").save()


def periodic_task():
    print('getting predicted data')
    ml_df_training = get_predicted_data()
    print('done')
    print('training models and saving to disk')
    kmeans_model = get_kmeans_model(ml_df_training)
    kmeans_model.write().overwrite().save('hdfs://10.8.0.11:9000/data/models/kmeans_model')
    rf_model = get_rf_model(ml_df_training)
    rf_model.write().overwrite().save('hdfs://10.8.0.11:9000/data/models/rf_model')
    print('done')


## get training data
train_features, train_labels, train_features_df, train_labels_df = get_training_data()

# create new dataframe for ml including labeling
ml_df_training = train_features_df.withColumn('label',
                                              F.when(F.col('dest_port') == 9200, 1).when(F.col('source_port') == 5600,
                                                                                         1).otherwise(0))

print('training initial models and saving to disk')
kmeans_model = get_kmeans_model(ml_df_training)
kmeans_model.write().overwrite().save('hdfs://10.8.0.11:9000/data/models/kmeans_model')
rf_model = get_rf_model(ml_df_training)
rf_model.write().overwrite().save('hdfs://10.8.0.11:9000/data/models/rf_model')
print('done')

# ml_df_live = get_live_data().withColumn('label', F.when(F.col('dest_port') == 9200, 1).when(F.col('source_port') == 5600, 1).otherwise(0))
# get_outliers(train_features, train_labels)

scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(periodic_task, "interval", minutes=5)

df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "10.8.0.8:9092").option(
    "kafkaConsumer.pollTimeoutMs", 10000).option("subscribe", "beats").load()

df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

# define schema of json
schema = StructType() \
    .add("@timestamp", StringType()) \
    .add("bytes_out", IntegerType()) \
    .add("ip", IntegerType()) \
    .add("client_ip", IntegerType()) \
    .add("dest", StructType() \
         .add('ip', StringType()) \
         .add('port', IntegerType())) \
    .add("source", StructType() \
         .add('ip', StringType()) \
         .add('port', IntegerType())) \
    .add("json", StructType() \
         .add('event_type', StringType()) \
         .add('src_ip', StringType()) \
         .add('dest_ip', StringType()) \
         .add('dns', StructType() \
              .add('rrname', StringType()) \
              .add('rrtype', StringType()))) \
 \
# apply json schema 
df = df.select(F.col("key").cast("string"), F.from_json(F.col("value").cast("string"), schema).alias('data'))
df.writeStream.foreachBatch(process_batch).start().awaitTermination()
