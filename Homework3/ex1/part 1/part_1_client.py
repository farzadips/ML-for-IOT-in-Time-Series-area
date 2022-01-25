import argparse
import requests
import tensorflow as tf
import base64


# parser = argparse.ArgumentParser()
# parser.add_argument('-command', nargs=1, type=str)
# parser.add_argument('-operands', nargs='+', type=float)
# args = parser.parse_args()


# command = args.command[0]
# operands = args.operands
name = "model.tflite"


fname = 'E:\Github\Machine-learning-for-IOT\Lab3\MLP.tflite'
cnn_bytes = bytearray (open(fname,'rb').read())

interpreter_encode =  base64.b64encode(cnn_bytes)
interpreter_string = interpreter_encode.decode()


url = 'http://localhost:8080'
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