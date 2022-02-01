import argparse

import requests

import tensorflow as tf

import base64

from DoSomething import DoSomething

import time

import json

from datetime import datetime






parser = argparse.ArgumentParser()

parser.add_argument('-path', nargs=1, type=str)
parser.add_argument('-name', nargs=1, type=str,required=False)
args = parser.parse_args()



if(args.path[0]=='add'):







    name = args.name[0]

    # operands = args.operands

    name = str(name) + ".tflite"





    fname = './' + name



    cnn_bytes = bytearray (open(fname,'rb').read())



    interpreter_encode =  base64.b64encode(cnn_bytes)

    interpreter_string = interpreter_encode.decode()





    url = 'http://192.168.184.132:8080/add'

    body = {

                        "mn": name,

                        "e": [

                            {"n": "model", "u": "/", "t": 0, "vd": interpreter_string}

                        ],

            }



    r = requests.put(url, json=body)

    if r.status_code == 200:

        print("done")

    else:

        print('Error:', r.status_code)



if(args.path[0]=='list'):

    url = 'http://192.168.184.132:8080/list'



    r = requests.get(url)





    if r.status_code == 200:

        body = r.json()

        print(body)

    else:

        print('Error:', r.status_code)

if(args.path[0] == 'predict'):

    test = DoSomething("publisher 1")

    test.run()



    #dht_device = adafruit_dht.DHT11(D4)





    body = {

            'model_name':args.name[0],

            'tth' : 0.1,

            'hth' : 0.2

    }

    body_json = json.dumps(body)

    test.myMqttClient.myPublish("/1362341525/tt", body_json)



    print("\n")



    test.end()



