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

app = Flask(__name__)
bootstrap = Bootstrap(app)

r = redis.StrictRedis(host='10.8.0.16', port=6379, db=0)

def ip2long(ip):
    """
    Convert an IP string to long
    """
    if ip:
        packedIP = socket.inet_aton(ip)
        return struct.unpack("!L", packedIP)[0]

def get_live_data():
    # pull live data set from es
    es = Elasticsearch(hosts="10.8.0.16")
    #s = Search(using=es, index="*").filter("term", type="flow").filter('range',
    #                                                                   **{'@timestamp': {'gte': 'now-1m', 'lt': 'now'}})
    s = Search(using=es, index="model_predictions*").filter('range', **{'@timestamp': {'gte': 'now-5m', 'lt': 'now'}})
    # s.aggs.bucket('by_timestamp', 'terms', field='@timestamp', size=999999999).metric('total_net_bytes', 'sum', field="source.stats.net_bytes_total")

    response = s.execute()

    data = []
    for hit in response:
        if 'port' in hit.source and 'ip' in hit.source:
            # print(ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port)
            id = uuid.uuid4().node
            data.append([id, hit.source.ip, hit.source.port, hit.dest.ip, hit.dest.port])
            r.set(id, json.dumps([id, hit.source.ip, hit.source.port, hit.dest.ip, hit.dest.port]))
    if data == []:
        data.append(['nodata','nodata','nodata','nodata','nodata'])
    print(data)
    # create dataframes for live data
    data = np.asarray(data)
    columns = ["id", "source_ip", "source_port", "dest_ip", "dest_port"]
    data_pddf = pd.DataFrame(data, columns=columns)
    return data_pddf


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
    for i in request.form:
        results.append(json.loads(r.get(i)))
    print(results)
    return json.dumps(results)


if __name__ == '__main__':
    app.run(debug=True)
