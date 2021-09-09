import argparse
import json
import os

import re

import nltk
from allennlp.predictors import Predictor

from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB


class Doc_Retrieval:

    def __init__(self, database_path, add_claim=False, k_wiki_results=None):
        self.db = FeverDocDB(database_path)
        self.add_claim = add_claim
        self.k_wiki_results = k_wiki_results
        self.proter_stemm = nltk.PorterStemmer()
        self.tokenizer = nltk.word_tokenize
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")    
    
    def get_tokens(self, line):

        claim = line['claim']
        tokens = self.predictor.predict(claim)
        
        return tokens

if __name__ == '__main__':

    home = os.getcwd()
    dataset_folder = os.path.join(home, "data")
    full_db = os.path.join(dataset_folder, "feverous-wiki-docs.db")
    raw_training_set = os.path.join(dataset_folder, "train.jsonl")
    training_output_file = os.path.join(dataset_folder, "train.output.jsonl")
    
    document_k_wiki = 7
    document_add_claim = True
    document_parallel = True
     
    jlr = JSONLineReader()
    lines = jlr.read(full_db)

    for i in range(len(lines)):
        np = Doc_Retrieval.get_tokens(line = lines[i])

#---