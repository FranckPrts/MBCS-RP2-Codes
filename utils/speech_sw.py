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

import numpy as np

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence


class read_dialogue:
    
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
        
        with open(self.path, 'r') as json_file:
            self.dialogue = json.load(json_file)

        # The JSON element that has all the words is the last
        self.all_word_dict_idx = len(self.dialogue["results"]) - 1

        # Make list with the woed spocken by each subject within
        # the given time window
        self.sub1_speech_list, self.sub2_speech_list = self.select_sw()

        # Embed each sentence
        self.sub1_speech_embed, self.sub2_speech_embed = self.embed_dialogue()

    def select_sw(self):

        sub1 = list()
        sub2 = list()

        for i in range(len(self.dialogue["results"][self.all_word_dict_idx]["alternatives"][0]["words"])):
            
            word = self.dialogue["results"][self.all_word_dict_idx]["alternatives"][0]["words"][i]

            if self.sw_min <= float(word["startTime"][:-1]) and float(word["endTime"][:-1]) <= self.sw_max:
                
                # Log in console
                print("[%f - %f] \tSpeacker: %s \t\tWord: %s" % (float(word["startTime"][:-1]), float(word["endTime"][:-1]), word["speakerTag"], word["word"]))

                # Save both speacker words in an array
                if word["speakerTag"] == 1:
                    sub1.append(word["word"])
                if word["speakerTag"] == 2:
                    sub2.append(word["word"])
        
        return sub1, sub2


    def embed_dialogue(self):
        sub1 = ' '.join(self.sub1_speech_list)
        sub2 = ' '.join(self.sub2_speech_list)

        embedding1 = TransformerWordEmbeddings()
        embedding2 = TransformerWordEmbeddings()

        sent1 = Sentence(sub1, use_tokenizer=True)
        sent2 = Sentence(sub2, use_tokenizer=True)

        embedding1.embed(sent1)
        embedding2.embed(sent2)

        sent1_em = np.array([s.embedding.cpu().numpy() for s in sent1])
        sent2_em = np.array([s.embedding.cpu().numpy() for s in sent2])

        return sent1_em, sent2_em

    def compute_semantic_similarity(self):
        # sent_avg = np.mean(sent1_em, axis=0)
        pass

    def count_turn_taking(self):
        pass