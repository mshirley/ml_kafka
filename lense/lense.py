import json

from flask import Flask, render_template, request, redirect
from flask_bootstrap import Bootstrap
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import numpy as np
import pandas as pd
import socket
import struct
import uuid
import redis
from pykafka import KafkaClient

app = Flask(__name__)
bootstrap = Bootstrap(app)
r = redis.StrictRedis(host="redis", port=6379, db=0)
k = KafkaClient(hosts="kafka:9092")
es = Elasticsearch(hosts="elasticsearch")


def ip2long(ip):
    """
    Convert ip to long

    :param ip:
    :return: long ip
    """
    if ip:
        packed_ip = socket.inet_aton(ip)
        return struct.unpack("!L", packed_ip)[0]


def get_live_data():
    """
    Query elasticsearch for live data

    :return: data pandas dataframe
    """
    s = Search(using=es, index="*").filter("term", type="flow").filter('range',
                                                                       **{'@timestamp': {'gte': 'now-1m', 'lt': 'now'}})
    response = s.execute()
    data = []
    for hit in response:
        if 'port' in hit.source and 'ip' in hit.source:
            id = uuid.uuid4().node
            values = [id, hit.source.ip, ip2long(hit.source.ip), hit.source.port, hit.dest.ip, ip2long(hit.dest.ip),
                      hit.dest.port]
            data.append(values)
            r.set(id, json.dumps(values))
    if data is []:
        data.append(['nodata', 'nodata', 'nodata', 'nodata', 'nodata', 'nodata', 'nodata'])
    print(data)
    data = np.asarray(data)
    columns = ["id", "source_ipv4", "source_ip", "source_port", "dest_ipv4", "dest_ip", "dest_port"]
    data_pddf = pd.DataFrame(data, columns=columns)
    return data_pddf


def get_prediction_data():
    """
    Query elasticsearch for prediction data

    :return: data pandas dataframe
    """
    s = Search(using=es, index="model_predictions-*").filter('term', algo='rf')\
        .filter('range', **{'@timestamp': {'gte': 'now-5m', 'lt': 'now'}})
    response = s.execute()
    data = []
    for hit in response:
        print(dir(hit))
        id = uuid.uuid4().node
        values = [id, hit.data.source.ip, hit.source_ip, hit.source_port, hit.data.dest.ip, hit.dest_ip,
                  hit.dest_port, hit.predictedLabel]
        data.append(values)
        r.set(id, json.dumps(values))
    if data is []:
        data.append(['nodata', 'nodata', 'nodata', 'nodata', 'nodata', 'nodata', 'nodata', 'nodata'])
    print(data)
    data = np.asarray(data)
    columns = ["id", "source_ipv4", "source_ip", "source_port", "dest_ipv4", "dest_ip", "dest_port", "predictedLabel"]
    data_pddf = pd.DataFrame(data, columns=columns)
    return data_pddf


def user_predictions_to_kafka(data):
    """
    send data to kafka

    :param data:
    """
    topic = k.topics['user_predictions']
    try:
        print(data)
        print(type(data))
        with topic.get_sync_producer() as producer:
            for d in data:
                producer.produce(bytes(json.dumps(d), 'utf8'))

    except Exception as e:
        print(e)
        pass


def validated_predictions_to_kafka(data):
    """
    send data to kafka

    :param data:
    """
    topic = k.topics['validated_predictions']
    try:
        print(data)
        print(type(data))
        with topic.get_sync_producer() as producer:
            for d in data:
                producer.produce(bytes(json.dumps(d), 'utf8'))

    except Exception as e:
        print(e)
        pass


@app.route('/')
def index():
    """
    main index route

    :return:
    """
    return render_template('index.html')


@app.route('/validate_predictions')
def validate_predictions():
    """

    :return:
    """
    prediction_data = get_prediction_data()
    print(prediction_data)
    return render_template('validate_predictions.html', prediction_data=prediction_data)


@app.route('/user_predictions')
def user_predictions():
    """

    :return:
    """
    live_data = get_live_data()
    print(live_data)
    return render_template('user_predictions.html', live_data=live_data)


@app.route('/submit_validated_predictions', methods=['POST', 'GET'])
def submit_validated_predictions():
    """
    submission route

    :return:
    """
    print(request.form)
    print(request.args)
    results = []
    print(request.form.keys())
    for k, v in request.form.items():
        print(k, v)
        data = json.loads(r.get(k))
        print(data)
        id = data[0]
        source_ipv4 = data[1]
        source_ip = data[2]
        source_port = data[3]
        dest_ipv4 = data[4]
        dest_ip = data[5]
        dest_port = data[6]
        if v == 'true':
            validated_label = 1
        else:
            validated_label = 0
        data = {'id': id,
                'source_ipv4': source_ipv4,
                'source_ip': source_ip,
                'source_port': source_port,
                'dest_ipv4': dest_ipv4,
                'dest_ip': dest_ip,
                'dest_port': dest_port,
                'validated_label': validated_label}
        results.append(data)
    print(results)
    validated_predictions_to_kafka(results)
    return redirect('/validate_predictions', code=302)


@app.route('/submit_user_predictions', methods=['POST', 'GET'])
def submit_user_predictions():
    """
    submission route

    :return:
    """
    print(request.form)
    print(request.args)
    results = []
    print(request.form.keys())
    for k, v in request.form.items():
        print(k, v)
        data = json.loads(r.get(k))
        print(data)
        id = data[0]
        source_ipv4 = data[1]
        source_ip = data[2]
        source_port = data[3]
        dest_ipv4 = data[4]
        dest_ip = data[5]
        dest_port = data[6]
        if v == 'true':
            label = 1
        else:
            label = 0
        data = {'id': id,
                'source_ipv4': source_ipv4,
                'source_ip': source_ip,
                'source_port': source_port,
                'dest_ipv4': dest_ipv4,
                'dest_ip': dest_ip,
                'dest_port': dest_port,
                'label': label}
        results.append(data)
    print(results)
    user_predictions_to_kafka(results)
    return redirect('/user_predictions', code=302)

if __name__ == '__main__':
    app.run(debug=True)
