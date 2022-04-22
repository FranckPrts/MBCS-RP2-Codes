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

    def __init__(self, path:str):
        
        self.path = path

        with open(self.path, 'r') as json_file:
            self.dialogue = json.load(json_file)

        # The JSON element that has all the words is the last
        self.all_word_dict_idx = len(self.dialogue["results"]) - 1

    def select_sw(self, sw_min:float, sw_max:float):
                
                for i in range(len(self.dialogue["results"][self.all_word_dict_idx]["alternatives"][0]["words"])):
                    
                    word = self.dialogue["results"][self.all_word_dict_idx]["alternatives"][0]["words"][i]                  
                    if sw_min <= float(word["startTime"][:-1]) and float(word["startTime"][:-1]) <= sw_max:
                        print(word["word"])



    def tokenize_dialogue(self):
        pass

    def compute_semantic_similarity(self):
        pass

    def count_turn_taking(self):
        pass