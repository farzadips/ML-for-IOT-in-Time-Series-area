import datetime
from DoSomething import DoSomething
import time
import json

import tensorflow as tf
import numpy as np



class Subscriber(DoSomething):
    def notify(self, topic, msg):
        print('entered notify')
        input_json = json.loads(msg)
        print(input_json['e']['v'])


if __name__ == "__main__":
    test = Subscriber("subscriber 10")
    test.run()
    test.myMqttClient.mySubscribe('/1362341525/alerts')
    print('started to subscribe alerts')

    while True:
        time.sleep(1)
