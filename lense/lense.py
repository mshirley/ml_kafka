from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
import numpy as np
import pandas as pd
import socket
import struct

app = Flask(__name__)
bootstrap = Bootstrap(app)

def ip2long(ip):
    """
    Convert an IP string to long
    """
    if ip:
        packedIP = socket.inet_aton(ip)
        return struct.unpack("!L", packedIP)[0]

def get_live_data():
    # pull live data set from es
    es = Elasticsearch(hosts="10.8.0.3")
    s = Search(using=es, index="*").filter("term", type="flow").filter('range',
                                                                       **{'@timestamp': {'gte': 'now-5m', 'lt': 'now'}})

    # s.aggs.bucket('by_timestamp', 'terms', field='@timestamp', size=999999999).metric('total_net_bytes', 'sum', field="source.stats.net_bytes_total")

    response = s.scan()

    data = []
    for hit in response:
        if 'port' in hit.source and 'ip' in hit.source:
            # print(ip2long(hit.source.ip), hit.source.port, ip2long(hit.dest.ip), hit.dest.port)
            data.append([hit.source.ip, hit.source.port, hit.dest.ip, hit.dest.port])

    # create dataframes for live data
    data = np.asarray(data)
    columns = ["source_ip", "source_port", "dest_ip", "dest_port"]
    data_pddf = pd.DataFrame(data, columns=columns)

    return data_pddf


# Setup Homepage Route
@app.route('/')
def index():
    live_data = get_live_data()
    print(live_data)
    return render_template('index.html', live_data=live_data)


if __name__ == '__main__':
    app.run(debug=True)