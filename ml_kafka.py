#!/usr/bin/env python
# coding: utf-8

import os
import socket
import struct
import sys

import numpy as np
import pandas as pd
import pyspark
from apscheduler.schedulers.background import BackgroundScheduler
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from pyspark import SQLContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.types import *

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    '--packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.1 pyspark-shell'

sc = pyspark.SparkContext(appName="ml_kafka")
sc.setLogLevel("ERROR")
spark = SQLContext(sc)

es = Elasticsearch(hosts="elasticsearch")


def get_kmeans_model(train_features_df):
    kmeans = KMeans(k=10, seed=1)
    model = kmeans.fit(train_features_df.select('features'))
    return model


def get_rf_model(train_features_df):
    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(train_features_df)
    feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(
        train_features_df)
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
    label_converter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                    labels=label_indexer.labels)
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, rf, label_converter])
    model = pipeline.fit(train_features_df)
    return model


def get_live_data():
    s = Search(using=es, index="*").filter("term", type="flow").filter('range',
                                                                       **{'@timestamp': {'gte': 'now-1m', 'lt': 'now'}})
    response = s.scan()
    feature_data = []
    for hit in response:
        if 'port' in hit.source and 'ip' in hit.source:
            feature_data.append([ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port])

    # create dataframes for live data
    data_array = np.asarray(feature_data)
    columns = ["source_ip", "source_port", "dest_ip", "dest_port"]
    data_pddf = pd.DataFrame(data_array, columns=columns)
    data_df = spark.createDataFrame(data_pddf)

    vec_assembler = VectorAssembler(inputCols=columns, outputCol="features")
    data_df = vec_assembler.transform(data_df)
    return data_df


def get_user_predictions():
    s = Search(using=es, index="user_predictions*").filter('range', **{'@timestamp': {'gte': 'now-1h', 'lt': 'now'}})
    response = s.scan()
    feature_data = []
    for hit in response:
        if "label" in hit:
            feature_data.append([int(hit.source_ip), int(hit.source_port), int(hit.dest_ip), int(hit.dest_port),
                                int(hit.label)])
    if feature_data == []:
        print('no user predictions available, create some using lense')
        sys.exit(1)
    else:
        columns = ["source_ip", "source_port", "dest_ip", "dest_port", "label"]
        feature_data = np.asarray(feature_data)
        data_pddf = pd.DataFrame(feature_data, columns=columns)
        data_df = spark.createDataFrame(data_pddf)
        feature_columns = ["source_ip", "source_port", "dest_ip", "dest_port"]
        vec_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data_df = vec_assembler.transform(data_df).select('source_ip', 'source_port', 'dest_ip', 'dest_port',
                                                          'features', 'label').withColumn('label',
                                                                                          F.col('label').cast('int'))
    return data_df


def ip2long(ip):
    """
    Convert an IP string to long
    """
    if ip:
        packed_ip = socket.inet_aton(ip)
        return struct.unpack("!L", packed_ip)[0]


ip2long_udf = F.udf(ip2long, LongType())


def normalize_ips(df):
    df = (df.withColumn('source_ip', F.col('data.json.src_ip'))
          .withColumn('dest_ip', F.col('data.json.dest_ip'))
          .withColumn('source_ip', F.col('data.client_ip'))
          .withColumn('dest_ip', F.col('data.ip'))
          .withColumn('dest_ip', F.col('data.dest.ip'))
          .withColumn('source_ip', F.col('data.source.ip'))
          .withColumn('dest_port', F.col('data.dest.port'))
          .withColumn('source_port', F.col('data.source.port'))
          .filter('source_ip is not NULL')
          .filter('dest_ip is not NULL')
          .filter('source_port is not NULL')
          .filter('dest_port is not NULL'))
    df = (df.withColumn('source_ip', ip2long_udf(F.col('source_ip')))
          .withColumn('dest_ip', ip2long_udf(F.col('dest_ip'))))
    return df


def process_batch(df, epoch_id):
    print(epoch_id)
    if epoch_id % 100 == 0:
        try:
            print('loading models')
            global kmeans_model
            kmeans_model = KMeansModel.load('hdfs://hdfs:9000/data/models/kmeans_model')
            global rf_model
            rf_model = PipelineModel.load('hdfs://hdfs:9000/data/models/rf_model')
        except Exception as e:
            print('unable to load modules, {}'.format(e))
    df = normalize_ips(df)
    columns = ["source_ip", "source_port", "dest_ip", "dest_port"]
    vec_assembler = VectorAssembler(inputCols=columns, outputCol="features")
    df = vec_assembler.transform(df)
    kmeans_model_result = kmeans_model.transform(df).withColumn('algo', F.lit('kmeans'))
    kmeans_model_result.selectExpr("CAST('key' AS STRING)", "to_json(struct(*)) AS value").write.format(
        "kafka").option("kafka.bootstrap.servers", "kafka:9092").option("topic", "model_predictions").save()

    rf_model_result = rf_model.transform(df).withColumn('algo', F.lit('rf'))
    rf_model_result.selectExpr("CAST('key' AS STRING)", "to_json(struct(*)) AS value").write.format(
        "kafka").option("kafka.bootstrap.servers", "kafka:9092").option("topic", "model_predictions").save()


def periodic_task():
    print('getting user predictions')
    new_user_predictions = get_user_predictions()
    print('done')
    df = spark.createDataFrame(sc.emptyRDD(), live_data_predictions.schema)
    data_df = df.union(new_user_predictions)
    print('training models and saving to disk')
    kmeans_model = get_kmeans_model(data_df)
    kmeans_model.write().overwrite().save('hdfs://hdfs:9000/data/models/kmeans_model')
    rf_model = get_rf_model(data_df)
    rf_model.write().overwrite().save('hdfs://hdfs:9000/data/models/rf_model')
    print('done')


def start_up():
    scheduler = BackgroundScheduler()
    scheduler.start()
    scheduler.add_job(periodic_task, "interval", minutes=5)
    df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "kafka:9092").option(
        "kafkaConsumer.pollTimeoutMs", 10000).option("subscribe", "beats").load()
    df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
    schema = StructType() \
        .add("@timestamp", StringType()) \
        .add("bytes_out", IntegerType()) \
        .add("ip", IntegerType()) \
        .add("client_ip", IntegerType()) \
        .add("dest", StructType()
             .add('ip', StringType())
             .add('port', IntegerType())) \
        .add("source", StructType()
             .add('ip', StringType())
             .add('port', IntegerType())) \
        .add("json", StructType()
             .add('event_type', StringType())
             .add('src_ip', StringType())
             .add('dest_ip', StringType())
             .add('dns', StructType()
                  .add('rrname', StringType())
                  .add('rrtype', StringType())))
    df = df.select(F.col("key").cast("string"), F.from_json(F.col("value").cast("string"), schema).alias('data'))
    df.writeStream.foreachBatch(process_batch).start().awaitTermination()


if __name__ == "__main__":  # get training data
    live_data = get_live_data()
    live_data_predictions = live_data.withColumn('label',
                                                 F.when(F.col('dest_port') == 9200, 1)
                                                 .when(F.col('source_port') == 9200, 1)
                                                 .when(F.col('dest_port') == 5601, 1)
                                                 .when(F.col('source_port') == 5601, 1)
                                                 .otherwise(0))
    user_predictions = get_user_predictions()
    empty_df = spark.createDataFrame(sc.emptyRDD(), live_data_predictions.schema)
    data = empty_df.union(user_predictions)
    print('training initial models and saving to disk')
    kmeans_model = get_kmeans_model(data)
    kmeans_model.write().overwrite().save('hdfs://hdfs:9000/data/models/kmeans_model')
    rf_model = get_rf_model(data)
    rf_model.write().overwrite().save('hdfs://hdfs:9000/data/models/rf_model')
    print('done')
    start_up()
