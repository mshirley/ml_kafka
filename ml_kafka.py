#!/usr/bin/env python
# coding: utf-8

import pyspark
from pyspark import SQLContext
from pyspark.sql.types import *
from pyspark.sql import functions as F
from apscheduler.schedulers.background import BackgroundScheduler
from data import *
from models import *
from misc import *

sc = pyspark.SparkContext(appName="ml_kafka")
sc.setLogLevel("ERROR")
spark = SQLContext(sc)

def normalize_ips(df):
    df = df.withColumn('source_ip', F.col('data.json.src_ip')).withColumn('dest_ip', F.col('data.json.dest_ip')).withColumn('source_ip', F.col('data.client_ip')).withColumn('dest_ip', F.col('data.ip')).withColumn('dest_ip', F.col('data.dest.ip')).withColumn('source_ip', F.col('data.source.ip')).withColumn('dest_port', F.col('data.dest.port')).withColumn('source_port', F.col('data.source.port')).filter('source_ip is not NULL').filter('dest_ip is not NULL').filter('source_port is not NULL').filter('dest_port is not NULL')
    df = df.withColumn('source_ip', ip2long_udf(F.col('source_ip'))).withColumn('dest_ip', ip2long_udf(F.col('dest_ip')))
    return(df)

def process_batch(df, epoch_id):
    print(epoch_id)
    if epoch_id % 100 == 0:
        try:
            print('loading models')
            KMeansModel.load('kmeans_model')
            PipelineModel.load('rf_model')
        except Exception as e:
            print('unable to load modules, {}'.format(e))
    df = normalize_ips(df)
    #df.show()
    columns = ["source_ip", "source_port", "dest_ip", "dest_port"]
    vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
    df = vecAssembler.transform(df)
    #print(df)
    kmeans_model_result = kmeans_model.transform(df).withColumn('algo', F.lit('kmeans'))
    #kmeans_model_result.show()
    ds = kmeans_model_result.selectExpr("CAST('key' AS STRING)", "to_json(struct(*)) AS value").write.format("kafka").option("kafka.bootstrap.servers", "10.8.0.8:9092").option("topic", "model_predictions").save()

    rf_model_result = rf_model.transform(df).withColumn('algo', F.lit('rf'))
    #rf_model_result.show()
    ds = rf_model_result.selectExpr("CAST('key' AS STRING)", "to_json(struct(*)) AS value").write.format("kafka").option("kafka.bootstrap.servers", "10.8.0.8:9092").option("topic", "model_predictions").save()

def periodic_task():
    print('getting predicted data')
    ml_df_training = get_predicted_data(spark)
    print('done')
    print('training models and saving to disk')
    kmeans_model = get_kmeans_model(ml_df_training)
    kmeans_model.write().overwrite().save('kmeans_model')
    rf_model = get_rf_model(ml_df_training)
    rf_model.write().overwrite().save('rf_model')
    print('done')

## get training data
train_features, train_labels, train_features_df, train_labels_df = get_training_data(spark)

# create new dataframe for ml including labeling
ml_df_training = train_features_df.withColumn('label', F.when(F.col('dest_port') == 9200, 1).when(F.col('source_port') == 5600, 1).otherwise(0))

print('training initial models and saving to disk')
kmeans_model = get_kmeans_model(ml_df_training)
kmeans_model.write().overwrite().save('kmeans_model')
rf_model = get_rf_model(ml_df_training)
rf_model.write().overwrite().save('rf_model')
print('done')

#ml_df_live = get_live_data().withColumn('label', F.when(F.col('dest_port') == 9200, 1).when(F.col('source_port') == 5600, 1).otherwise(0))
#get_outliers(train_features, train_labels)

scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(periodic_task, "interval", minutes=5)

df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "10.8.0.8:9092").option("kafkaConsumer.pollTimeoutMs", 5000).option("subscribe", "logs").load()

df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

# define schema of json
schema = StructType() \
        .add("@timestamp", StringType()) \
        .add("bytes_out", IntegerType()) \
        .add("ip", IntegerType()) \
        .add("client_ip", IntegerType()) \
        .add("dest", StructType() \
        .add('ip', StringType())\
        .add('port', IntegerType()))\
        .add("source", StructType() \
            .add('ip', StringType()) \
            .add('port', IntegerType()))\
        .add("json", StructType() \
            .add('event_type', StringType()) \
            .add('src_ip', StringType()) \
            .add('dest_ip', StringType()) \
            .add('dns', StructType() \
                .add('rrname', StringType()) \
                .add('rrtype', StringType()))) \
# apply json schema 
df = df.select(F.col("key").cast("string"), F.from_json(F.col("value").cast("string"), schema).alias('data'))
df.writeStream.foreachBatch(process_batch).start().awaitTermination()
