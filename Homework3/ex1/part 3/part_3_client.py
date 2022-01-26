import argparse
import requests


parser = argparse.ArgumentParser()
parser.add_argument('-m', nargs=1, type=str)
parser.add_argument('-tt', nargs=1, type=float)
parser.add_argument('-ht', nargs=1, type=float)
args = parser.parse_args()


model = args.m[0]
tthresh = args.tt[0]
hthresh = args.ht[0]


url = 'http://localhost:8080/predict?m={}&th={}&hh={}'.format(model, tthresh, hthresh)

r = requests.get(url)



if r.status_code == 200:
    body = r.json()
    print(body['result'])

else:
    print('Error:', r.status_code)
