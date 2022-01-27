from DoSomething import DoSomething
import time
import json
from datetime import datetime

from board import D4
import adafruit_dht


if __name__ == "__main__":
    test = DoSomething("publisher 1")
    test.run()

    dht_device = adafruit_dht.DHT11(D4)

    # available for 10 minutes

        body = {
				"bn": "http://192.168.1.9/",
                "model_name": "cnn",
                "tth" : 0.2,
                "hth" : 0.1
        }
        body_json = json.dumps(body)
        test.myMqttClient.myPublish("/1362341525/tt", body_json)

        print("\n")

    test.end()
