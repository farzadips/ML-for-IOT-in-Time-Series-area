import argparse
import requests
import time


parser = argparse.ArgumentParser()
parser.add_argument('d', nargs=1, type=float)
parser.add_argument('f', nargs=1, type=float)
args = parser.parse_args()

d = args.d[0]
f = args.f[0]
t = 25
h = 68
for i in range(int(d/f)):
    url = 'http://localhost:8080/?t={}&h={}'.format(t,h)
    r = requests.get(url)
    if r.status_code == 200:
        print('sent')
    else:
        print("error:",  r.status_code)
    h+=1
    t+=1
    time.sleep(f)
