#import adafruit_dht
import argparse
import datetime
import time
#from board import D4
import tensorflow as tf
import numpy as np
import IPython.display as display
import os
#import struct



parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help='input filename')
parser.add_argument('-o', type=str, help='output filename')
parser.add_argument('-n', type=str, help='normalize',default="n")

args = parser.parse_args()

MIN_TEMP= 0
MAX_TEMP = 50
MIN_HUM = 20
MAX_HUM = 90
filename = args.o

def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def computation():
    with tf.io.TFRecordWriter(filename) as writer:
        with open(args.i, 'r') as f:

            data_line = f.readline()

            while data_line != "" and data_line != None:
                data_line = data_line.split(',')
                sdate = data_line[0] + ' ' + data_line[1]
                datetimeobj=datetime.datetime.strptime(sdate,"%d/%m/%Y %H:%M:%S")
                timeobj = time.mktime(datetimeobj.timetuple())
                if (args.n=='y'):
                    temp = (float(data_line[2]) - MIN_TEMP)/ (MAX_TEMP - MIN_TEMP)
                    hum = (float(data_line[3]) - MIN_HUM)/(MAX_HUM - MIN_HUM)
                elif(args.n=='n'):
                    temp = float(data_line[2])
                    hum = float(data_line[3])

                mapping = { 
                        'datetime': _float_feature((timeobj)), 
                        'temperature': _int64_feature(int(temp)), 
                        'humidity': _int64_feature(int(hum))
                    }
                example = tf.train.Example(features=tf.train.Features(feature=mapping))
                writer.write(example.SerializeToString())
                data_line = f.readline()
                
if __name__ == '__main__':
    computation()
    print(f'csv: {os.path.getsize(args.i)}')
    print(f'tfrecord: {os.path.getsize(filename)}')