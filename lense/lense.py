import json

from flask import Flask, render_template, request
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
topic = k.topics['user_predictions']


def ip2long(ip):
    """
    Convert an IP string to long
    """
    if ip:
        packed_ip = socket.inet_aton(ip)
        return struct.unpack("!L", packed_ip)[0]


def get_live_data():
    # pull live data set from es
    es = Elasticsearch(hosts="elasticsearch")
    s = Search(using=es, index="*").filter("term", type="flow").filter('range',
                                                                       **{'@timestamp': {'gte': 'now-1m', 'lt': 'now'}})

    # s.aggs.bucket('by_timestamp', 'terms', field='@timestamp', size=999999999).metric('total_net_bytes', 'sum', field="source.stats.net_bytes_total")

    response = s.execute()

    data = []
    for hit in response:
        if 'port' in hit.source and 'ip' in hit.source:
            # print(hit.source.ip, ip2long(hit.source.ip), hit.source.port, hit.dest.ip, ip2long(hit.dest.ip), hit.dest.port)
            id = uuid.uuid4().node
            data.append([id, hit.source.ip, ip2long(hit.source.ip), hit.source.port, hit.dest.ip, ip2long(hit.dest.ip), hit.dest.port])
            #data.append([id, hit.source.ip, hit.source.port, hit.dest.ip, hit.dest.port])
            r.set(id, json.dumps([id, hit.source.ip, ip2long(hit.source.ip), hit.source.port, hit.dest.ip, ip2long(hit.dest.ip), hit.dest.port]))
    if data is []:
        data.append(['nodata','nodata', 'nodata','nodata','nodata','nodata','nodata'])
    print(data)
    # create dataframes for live data
    data = np.asarray(data)
    columns = ["id", "source_ipv4", "source_ip", "source_port", "dest_ipv4", "dest_ip", "dest_port"]
    data_pddf = pd.DataFrame(data, columns=columns)
    return data_pddf


def to_kafka(data):
    try:
        print(data)
        print(type(data))
        with topic.get_sync_producer() as producer:
            for d in data:
                producer.produce(bytes(json.dumps(d), 'utf8'))

    except Exception as e:
        print(e)
        pass


# Setup Homepage Route
@app.route('/')
def index():
    live_data = get_live_data()
    print(live_data)
    return render_template('index.html', live_data=live_data)


@app.route('/submit', methods=['POST', 'GET'])
def submit():
    print(request.form)
    print(request.args)
    results = []
    print(request.form.keys())
    for k, v in request.form.items():
        print(k,v)
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
    to_kafka(results)
    return json.dumps(results)


if __name__ == '__main__':
    app.run(debug=True)
