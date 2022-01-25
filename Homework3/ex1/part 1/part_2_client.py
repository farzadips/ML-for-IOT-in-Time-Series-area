import argparse
import requests



#this line didn't work
#url = 'http://0.0.0.0:8080/{}?op1={}&op2={}'.format(command, op1, op2)
url = 'http://localhost:8080'



r = requests.get(url)


if r.status_code == 200:
    body = r.json()
    print(body)
else:
    print('Error:', r.status_code)
