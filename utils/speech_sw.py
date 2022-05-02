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
# from cv2 import mean

import numpy as np

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
from sympy import EX
from transformers import PrinterCallback

import gensim.downloader as api

# Compute cosine disctance between 1-D vector https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html?highlight=cosine#scipy.spatial.distance.cosine
from scipy import spatial

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

    def __init__(self, path:str):
        
        self.path = path
        
        with open(self.path, 'r') as json_file:
            self.dialogue = json.load(json_file)

        # The JSON element that has all the words is the last
        self.all_word_dict_idx = len(self.dialogue["results"]) - 1

    def select_sw(self, sw_min:float, sw_max:float):

        sub1 = list()
        sub2 = list()

        print(">> Using %f %f time window\nStarting dialogue extraction ..." % (sw_min, sw_max))

        for i in range(len(self.dialogue["results"][self.all_word_dict_idx]["alternatives"][0]["words"])):
            
            word = self.dialogue["results"][self.all_word_dict_idx]["alternatives"][0]["words"][i]

            if sw_min <= float(word["startTime"][:-1]) and float(word["endTime"][:-1]) <= sw_max:
                
                # Log in console
                print("[%f - %f] \tSpeacker: %s \t\tWord: %s" % (float(word["startTime"][:-1]), float(word["endTime"][:-1]), word["speakerTag"], word["word"]))

                # Save both speacker words in an array
                if word["speakerTag"] == 1:
                    sub1.append(word["word"])
                if word["speakerTag"] == 2:
                    sub2.append(word["word"])
        
        self.sub1_speech_list, self.sub2_speech_list = sub1, sub2
        print(">> Dialogue extracted.")


    def embed_dialogue(self):

        print(">> Starting dialogue embeding ...")
       
        wtov = api.load('word2vec-google-news-300')
        
        s1 = list()
        s2 = list()
        
        print(">> Embeding S1 ...")
        for  word in self.sub1_speech_list:
            try:
                print(word)
                s1.append(wtov[word])
            except Exception as e :
                print(e)
                pass    
        print(">> Done.")

        print(">> Embeding S2 ...")
        for  word in self.sub2_speech_list:
            try:
                print(word)
                s2.append(wtov[word])
            except Exception as e :
                print(e)
                pass    
        print(">> Done.")

        # sub1 = ' '.join(self.sub1_speech_list)
        # sub2 = ' '.join(self.sub2_speech_list)

        # embedding1 = TransformerWordEmbeddings()
        # embedding2 = TransformerWordEmbeddings()

        # sent1 = Sentence(sub1, use_tokenizer=True)
        # sent2 = Sentence(sub2, use_tokenizer=True)

        # embedding1.embed(sent1)
        # embedding2.embed(sent2)

        # sent1_em = np.array([s.embedding.cpu().numpy() for s in sent1])
        # sent2_em = np.array([s.embedding.cpu().numpy() for s in sent2])

        self.sub1_speech_embed, self.sub2_speech_embed = s1, s2
        print(">> Dialogue embeded.")

    def compute_semantic_similarity(self):
        
        # Average all the words of a sentence 
        self.s1_speech_embed_avr = np.mean(self.sub1_speech_embed, axis=0)
        self.s2_speech_embed_avr = np.mean(self.sub2_speech_embed, axis=0)
        
        self.sem_sim = 1 - spatial.distance.cosine(self.s1_speech_embed_avr, self.s2_speech_embed_avr)

    def count_turn_taking(self):
        pass