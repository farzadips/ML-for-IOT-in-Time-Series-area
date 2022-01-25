import cherrypy
import json
from functools import reduce
import base64


class Calculator(object):
    exposed = True

    def GET(self, *path, **query):
        from os import listdir
        from os.path import isfile, join
        onlyfiles = [f for f in listdir('E:\Github\Machine-learning-for-IOT\Homework3\ex1\part 1\models') if isfile(join('E:\Github\Machine-learning-for-IOT\Homework3\ex1\part 1\models', f))]
        return onlyfiles
    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        print("recieved")
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        body = cherrypy.request.body.read()
        body = json.loads(body)

        name = body['mn']
        events = body['e']
        for event in events:
                model_string = event['vd'] 
        model_bytes = base64.b64decode(model_string)


        tflite_model_dir = "E:/Github/Machine-learning-for-IOT/Homework3/ex1/part 1/models/" + name
        with open(tflite_model_dir, 'wb') as fp:
            fp.write(model_bytes)
        output = {
                    'result': "done",
                }
        output_json = json.dumps(output)
        return output_json

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Calculator(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
