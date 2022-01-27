import datetime
from DoSomething import DoSomething
import time
import json

import tensorflow as tf
import numpy as np



class Subscriber(DoSomething):
    def notify(self, topic, msg):

        input_json = json.loads(msg)
        for event in input_json['e']:
            print(event['v'])


if __name__ == "__main__":
    test = Subscriber("subscriber 2")
    test.run()
    test.myMqttClient.mySubscribe("/1362341525/alerts")

    while True:
        time.sleep(1)
