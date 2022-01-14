import cherrypy
import json


class Calculator(object):
    exposed = True

    def GET(self, *path, **query):

        if len(path) != 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) != 2:
            raise cherrypy.HTTPError(400, 'Wrong query')


        t = query.get('t')
        if t is None:
            raise cherrypy.HTTPError(400, 't missing')
        else:
            t = float(t)

        h = query.get('h')
        if h is None:
            raise cherrypy.HTTPError(400, 'h missing')
        else:
            h = float(h)

        return str(t)+ "," +str(h)

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Calculator(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
