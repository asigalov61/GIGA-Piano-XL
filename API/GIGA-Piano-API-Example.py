# DO NOT FORGET TO ALLOW python.exe through Firewall
# Do not forget to change model settings and path to the desired ones

# Request to the API can be sent like so:
# !curl --header "Content-Type: application/json" \
#  --request POST \
#  --data '{"note_string":[[10, 8, 60], [12, 10, 62]],"note_count":10}' \
#  http://65.19.132.172:6000/

from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from waitress import serve

import os
import random
import copy

from collections import OrderedDict

from tqdm import tqdm

import numpy as np

import TMIDIX
from GPT2RGAX import *

app = Flask(__name__)
api = Api(app)

config = GPTConfig(128, 
                    1024,
                    dim_feedforward=1024,
                    n_layer=16, 
                    n_head=16, 
                    n_embd=1024,
                    enable_rpr=True,
                    er_len=1024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT(config)

state_dict = torch.load('C:/Users/paperspace/Desktop/GP/GIGA_Piano_Trained_Model_120000_steps_0.8075_loss.pth', map_location=device)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] #remove 'module'
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

model.to(device)

model.eval()

print('Ready')

# argument parsing

parser = reqparse.RequestParser()
parser.add_argument('note_string', type=str, help='Note string like 127-10-8-60-127-12-10-62-127-15-12-64')
parser.add_argument('note_count', type=int, help='Number of notes to generate')

class PredictSentiment(Resource):
    def post(self):

        json_data = request.get_json(force=True)
        user_query = json_data['note_string']
        note_count = json_data['note_count']

        print(user_query)

        inp = []
        for u in user_query:
          inp.extend([127])
          inp.extend(u)
        
        print(inp)

        t_count = (int(note_count)*4) + len(inp)

        rand_seq = model.generate_batches(torch.Tensor(inp), 
                                          target_seq_length=t_count,
                                          temperature=0.8,
                                          num_batches=1,
                                          verbose=False)
  
        out1 = rand_seq[0].cpu().tolist()

        in_notes = user_query
        
        note=[]

        out_notes = []

        for o in out1[-(int(note_count)*4):]:
           if o != 127:
             note.append(o)
           else:
             if len(note) > 0:
               out_notes.append(note)
             note = []
        
        return jsonify(input_notes=in_notes, output_notes=out_notes)


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


# driver function
if __name__ == '__main__':
  serve(app, port=6000)
