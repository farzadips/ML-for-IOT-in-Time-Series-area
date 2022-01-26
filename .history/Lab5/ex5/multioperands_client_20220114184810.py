import argparse
from sqlite3 import Timestamp
import requests
import base64
import json
import wave
import datatime

url = 'http://rapberrypi.local:8080'

for i in range(3):
    r = requests.get(url)

    if r.status_code == 200:
        body = r.json()
    
        timestamp = body['bt']
        events = body['e']
        for event in events:
            if event['n'] == 'temperature':
                temperature = event['v']
                t_unit = event['u']
            elif event['n'] == 'humidity':
                humidity = event['v']
                h_unit = event['u']
            elif event['n'] == 'audio':
                audio_string = event['vd'] 
        date = datatime.datatime.fromtimestamp(timestamp)
        audio_bytes = base64.b64decode(audio_string)

        print('{:02}/{:02}/{:04} {:02}:{:02}:{:02} {}{} {}{}'.format(
            date.day , date.month,date.year, date.hour ,date.minute,date.second,
            temperature,t_unit,humidity,h_unit
        ))

        wav_output_filename = "{}.wav".format(timestamp)
        wavefile = wave.open(wav_output_filename,'wb')
        wavefile.setchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(48000)
        wavefile.close()
    else:
        print('Error: ', r.status_code)
        break

    time.sleep(30)
    