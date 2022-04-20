#!/usr/bin/env python
# coding=utf-8

'''
Tools to import an speech transcription (dialogues in JSON format),
select content falling in a defined time window (sw), concatenate
then tokenize the items using TransformerWordEmbeddings, compute
the average semantic similarity of each side of the dialogue, and
returns the difference.
'''

# Imports 
import json

class read_dialogue ():
    
    """
    Implementing semantic similarity computation between each contribution
    to a dialogue. 

    Arguments:
        path: path
            Path to SET file (ablsolute or relative)
        sw_min: float
            Minimum boudary for the sliding window 
        sw_mxn: float
            Maximum boudary for the sliding window
    """

    def __init__(self, path:str, sw_min:float, sw_max:float):
        
        self.path = path
        self.sw_min = sw_min
        self.sw_max = sw_max

        # Import JSON https://www.freecodecamp.org/news/python-parse-json-how-to-read-a-json-file/
        with open(self.path, 'r') as json_file:
            
            # Get appropriate func
            self.dialogue = json.loads(json_file)
            # self.dialogue = json.loads(json_file)
            print(self.dialogue)

    def select_sw(self):
        pass

    def tokenize_dialogue(self):
        pass

    def compute_semantic_similarity(self):
        pass

    def count_turn_taking(self):
        pass