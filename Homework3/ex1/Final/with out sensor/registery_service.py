import cherrypy
import json
from functools import reduce
import base64
import os
import datetime
from DoSomething import DoSomething
import time
import json

import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np
import time


class Calculator1(object):
    exposed = True

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        body = cherrypy.request.body.read()
        body = json.loads(body)
        if(str(body['mn']) == ""):
            raise cherrypy.HTTPError(400, 'empty model name')
        if(str(body['mn']).find(".tflite") == -1):
            raise cherrypy.HTTPError(400, 'wrong name format')
        if(len(body['e']) == 0):
            raise cherrypy.HTTPError(400, 'no model file sent')
        name = body['mn']
        events = body['e']
        for event in events:
                model_string = event['vd'] 
        model_bytes = base64.b64decode(model_string)
        
        if not os.path.exists('./models'):      
                os.mkdir('models')

        tflite_model_dir = "./models/" + name
        with open(tflite_model_dir, 'wb') as fp:
            fp.write(model_bytes)
        output = {
                    'result': "done",
                }
        output_json = json.dumps(output)
        return output_json

    def DELETE(self, *path, **query):
        pass


#####################

class Calculator2(object):
    exposed = True

    def GET(self, *path, **query):
        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')
        from os import listdir
        from os.path import isfile, join

        onlyfiles = [f for f in listdir('./models') if isfile(join('./models', f))]
        print(onlyfiles)
        verified = False 
        if(len(onlyfiles)==2):
            verified = True
        output = {'path': onlyfiles, 'verified' : verified}
        output_json = json.dumps(output)

        #return onlyfiles
        return output_json






    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        
       
        
        pass

    def DELETE(self, *path, **query):
        pass


######################

class Subscriber_and_publish(DoSomething):
    def notify(self, topic, msg):

        #dht_device = adafruit_dht.DHT11(D4)
        if len(msg) == 0:
            raise cherrypy.HTTPError(400, 'no data recieved')
        input_json = json.loads(msg)

        if(str(input_json['model_name']) == ""):
            raise cherrypy.HTTPError(400, 'empty model name')

        model_name = input_json['model_name']
        if(len(input_json['e']) <2):
            raise cherrypy.HTTPError(400, 'not enough tresh holds indicated')

        for trs in input_json['e']:
            if threshes['n'] == 'temperature':
                tt = threshes['v']
            else:
                th = threshes['v']

        model_name = str(model_name) + ".tflite"
        model_path = './models/' + model_name
        if(os.path.exists(model_path) == False):
            raise cherrypy.HTTPError(400, 'model not found')
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        temp_hum_set = np.array([[]],dtype=np.float32)
        first = True
        window = np.zeros([1, 6, 2], dtype=np.float32)
        expected = np.zeros(2, dtype=np.float32)

        MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
        STD = np.array([ 8.654227, 16.557089], dtype=np.float32)
        #dht_device = adafruit_dht.DHT11(D4)
        while(True):
            if(first):
                # temprature:
                #temp = dht_device.temprature
                # humidity:
                #hum = dht_device.humidity
                temp_hum_set = np.array([[-8, 93]],dtype=np.float32)
                first = False
            else:
            #np.array([[[-8.2, 93.3],[-8.41, 93.4],[-8.51, 93.90],[-8.31, 94.2],[-8.27, 94.1],[-8.05, 94.4],[-7.62, 94.8]]],dtype=np.float32)
                # temprature:
                #temp = dht_device.temprature
                # humidity:
                #hum = dht_device.humidity
                temp_hum_set = np.append(temp_hum_set,[[-8, 93]], axis = 0)
            
            if(temp_hum_set.shape[0]>6):
                print("predicting")
                datetime_str = str(now.strftime('%d-%m-%y %H:%M:%S'))
                for i in range(7):
                    temperature = temp_hum_set[i][0]
                    humidity = temp_hum_set[i][1]
                    if i < 6:
                        window[0, i, 0] = np.float32(temperature)
                        window[0, i, 1] = np.float32(humidity)
                    if i == 6:
                        expected[0] = np.float32(temperature)
                        expected[1] = np.float32(humidity)
                        window = (window - MEAN) / STD
                        interpreter.set_tensor(input_details[0]['index'], window)
                        interpreter.invoke()
                        predicted = interpreter.get_tensor(output_details[0]['index'])
                        temp_hum_set = temp_hum_set[1:]
                        if(abs(predicted[0][0][0]-expected[0])>tt):
                            error_temp =  '(' + str(datetime_str) + ')' +'  Temperature Alert: Predicted=' + str(predicted[0][0][0]) + '°C Actual=' + str(expected[0]) + '°C'
                            
                            error_temp = {"n": "temperature", "u":"/", "t":0, "v":error_temp} 
                            
                            body = {
                                    "e": error_temp
                            }
                            body_json = json.dumps(body)
                            
                            print("Alert published for Temperature")
                            test2.myMqttClient.myPublish("/1352341525/alerts", body_json)
                            #return output_json
                        if(abs(predicted[0][0][1]-expected[1])>th):
                            error_humd = '(' + str(datetime_str) + ')' + '  Humidity Alert: Predicted=' + str(predicted[0][0][1]) + '% Actual=' + str(expected[1]) + '%'
                            
                            
                            error_humd= {"n": "humidity", "u":"/", "t":0, "v":error_humd}
                            body = {
                                    "e": error_humd
                            }
                            body_json = json.dumps(body)
                            print("Alert published for Humidity")
                            test2.myMqttClient.myPublish("/1352341525/alerts", body_json)

                          

        print("\n")

        test.end()
        test2.end()
                



if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Calculator1(), '/add', conf)
    cherrypy.tree.mount(Calculator2(), '/list', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    

    test = Subscriber_and_publish("subscriber 1")
    test.run()
    test.myMqttClient.mySubscribe("/1362341525/tt")
    
    test2 = Subscriber_and_publish("publisher 1")
    test2.run()
    cherrypy.engine.block()
