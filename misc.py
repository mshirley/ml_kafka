from pyspark.sql import functions as F
from pyspark.sql.types import *
import socket, struct

def ip2long(ip):
    """
    Convert an IP string to long
    """
    if ip:
        packedIP = socket.inet_aton(ip)
        return struct.unpack("!L", packedIP)[0]
ip2long_udf = F.udf(ip2long, LongType())

