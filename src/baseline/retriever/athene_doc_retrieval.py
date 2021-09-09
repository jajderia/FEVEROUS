#source: https://github.com/UKPLab/fever-2018-team-athene/blob/master/src/athene/retrieval/document/docment_retrieval.py

#call:
#document_retrieval_main(Config.db_path, Config.document_k_wiki, Config.raw_training_set,
#                                Config.training_doc_file,
#                                Config.document_add_claim, Config.document_parallel)

import argparse
import json
import os
from enum import Enum

import re
import time
from multiprocessing.pool import ThreadPool

from utils.log_helper import LogHelper

import nltk
import wikipedia
from allennlp.predictors import Predictor
from drqa.retriever.utils import normalize
from tqdm import tqdm

from common.dataset.reader import JSONLineReader
from retrieval.fever_doc_db import FeverDocDB

def processed_line(method, line):
    nps, wiki_results, pages = method.exact_match(line)
    line['noun_phrases'] = nps
    line['predicted_pages'] = pages
    line['wiki_results'] = wiki_results
    return line


def process_line_with_progress(method, line, progress=None):
    if progress is not None and line['id'] in progress:
        return progress[line['id']]
    else:
        return processed_line(method, line)


class Doc_Retrieval:

    def __init__(self, database_path, add_claim=False, k_wiki_results=None):
        self.db = FeverDocDB(database_path)
        self.add_claim = add_claim
        self.k_wiki_results = k_wiki_results
        self.proter_stemm = nltk.PorterStemmer()
        self.tokenizer = nltk.word_tokenize
        self.predictor = Predictor.from_path(
            #"https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
        
    def get_NP(self, tree, nps):

        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    # print(tree)
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    nps.append(tree['word'])
                    self.get_NP(tree['children'], nps)
                else:
                    self.get_NP(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                self.get_NP(sub_tree, nps)

        return nps

    def get_subjects(self, tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])
        return subjects

    def get_noun_phrases(self, line):

        claim = line['claim']
        tokens = self.predictor.predict(claim)
        nps = []
        tree = tokens['hierplane_tree']['root']
        noun_phrases = self.get_NP(tree, nps)
        subjects = self.get_subjects(tree)
        for subject in subjects:
            if len(subject) > 0:
                noun_phrases.append(subject)
        if self.add_claim:
            noun_phrases.append(claim)
        return list(set(noun_phrases))

    def get_doc_for_claim(self, noun_phrases):

        predicted_pages = []
        for np in noun_phrases:
            if len(np) > 300:
                continue
            i = 1
            while i < 12:
                try:
                    docs = wikipedia.search(np)
                    if self.k_wiki_results is not None:
                        predicted_pages.extend(docs[:self.k_wiki_results])
                    else:
                        predicted_pages.extend(docs)
                except (ConnectionResetError, ConnectionError, ConnectionAbortedError, ConnectionRefusedError):
                    print("Connection reset error received! Trial #" + str(i))
                    time.sleep(600 * i)
                    i += 1
                else:
                    break

            # sleep_num = random.uniform(0.1,0.7)
            # time.sleep(sleep_num)
        predicted_pages = set(predicted_pages)
        processed_pages = []
        for page in predicted_pages:
            page = page.replace(" ", "_")
            page = page.replace("(", "-LRB-")
            page = page.replace(")", "-RRB-")
            page = page.replace(":", "-COLON-")
            processed_pages.append(page)

        return processed_pages

    def np_conc(self, noun_phrases):

        noun_phrases = set(noun_phrases)
        predicted_pages = []
        for np in noun_phrases:
            page = np.replace('( ', '-LRB-')
            page = page.replace(' )', '-RRB-')
            page = page.replace(' - ', '-')
            page = page.replace(' :', '-COLON-')
            page = page.replace(' ,', ',')
            page = page.replace(" 's", "'s")
            page = page.replace(' ', '_')

            if len(page) < 1:
                continue
            doc_lines = self.db.get_doc_lines(page) """Fetch the raw text of the doc for 'page'."""
            if doc_lines is not None:
                predicted_pages.append(page) #just use to check if there are entries, lose doc text afterwards
        return predicted_pages

    def exact_match(self, line):

        noun_phrases = self.get_noun_phrases(line)
        wiki_results = self.get_doc_for_claim(noun_phrases)
        wiki_results = list(set(wiki_results))

        claim = normalize(line['claim'])
        claim = claim.replace(".", "")
        claim = claim.replace("-", " ")
        words = [self.proter_stemm.stem(word.lower()) for word in self.tokenizer(claim)]
        words = set(words)
        predicted_pages = self.np_conc(noun_phrases)

        for page in wiki_results:
            page = normalize(page)
            processed_page = re.sub("-LRB-.*?-RRB-", "", page)
            processed_page = re.sub("_", " ", processed_page)
            processed_page = re.sub("-COLON-", ":", processed_page)
            processed_page = processed_page.replace("-", " ")
            processed_page = processed_page.replace("–", " ")
            processed_page = processed_page.replace(".", "")
            page_words = [self.proter_stemm.stem(word.lower()) for word in self.tokenizer(processed_page) if
                          len(word) > 0]

            if all([item in words for item in page_words]):
                if ':' in page:
                    page = page.replace(":", "-COLON-")
                predicted_pages.append(page)
        predicted_pages = list(set(predicted_pages))
        # print("claim: ",claim)
        # print("nps: ",noun_phrases)
        # print("wiki_results: ",wiki_results)
        # print("predicted_pages: ",predicted_pages)
        # print("evidence:",line['evidence'])
        
        annotation_processor = AnnotationProcessor("{}/{}.jsonl".format(args.data_path, args.split))
        
        for page in predicted_pages:
            
        
        return noun_phrases, wiki_results, predicted_pages, annotations


def get_map_function(parallel, p=None):
    assert not parallel or p is not None, "A ThreadPool object should be given if parallel is True"
    return p.imap_unordered if parallel else map

def document_retrieval_main(db_file, k_wiki, in_file, out_file, add_claim=True, parallel=True):
    # tfidf_path = "data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"
    method = Doc_Retrieval(database_path=db_file, add_claim=add_claim, k_wiki_results=k_wiki)
    processed = dict()
    path = os.getcwd()
    jlr = JSONLineReader()
    lines = jlr.read(os.path.join(path, in_file))
    if os.path.isfile(os.path.join(path, in_file + ".progress")):
        with open(os.path.join(path, in_file + ".progress"), 'rb') as f_progress:
            import pickle
            progress = pickle.load(f_progress)
            print(os.path.join(path, in_file + ".progress") + " exists. Load it as progress file.")
    else:
        progress = dict()

    try:
        with ThreadPool(processes=4 if parallel else None) as p:
            # for line in tqdm( [thing1][thing2], [thing3])
            for line in tqdm(
                    get_map_function(parallel, p)
                     (lambda l: process_line_with_progress(method, l, progress), lines),
                    total=len(lines)):
                processed[line['id']] = line
                progress[line['id']] = line
                # time.sleep(0.5)
        with open(os.path.join(path, out_file), "w+") as f2:
            for line in lines:
                f2.write(json.dumps(processed[line['id']]) + "\n") #just a list of ids!!!
    finally:
        with open(os.path.join(path, in_file + ".progress"), 'wb') as f_progress:
            import pickle
            pickle.dump(progress, f_progress, pickle.HIGHEST_PROTOCOL)



#------

import os.path as path

BASE_DIR = os.getcwd()
dataset_folder = path.join(BASE_DIR, "data")

db_path = path.join(dataset_folder, "feverous-wiki-docs.db")

#dataset_folder = path.join(BASE_DIR, "data/fever")
#db_path = path.join(BASE_DIR, "data/fever/fever.db")
document_k_wiki = 7

document_add_claim = True
document_parallel = True

#input file
raw_training_set = path.join(dataset_folder, "train.jsonl")
raw_dev_set = path.join(dataset_folder, "dev.jsonl")
#raw_test_set = path.join(BASE_DIR, "data/fever-data/test.jsonl")

#output file path
training_doc_file = path.join(dataset_folder, "train.output.jsonl")
dev_doc_file = path.join(dataset_folder, "dev.output.jsonl")
#test_doc_file = path.join(dataset_folder, "test.wiki7.jsonl")

def document_retrieval(logger):
    logger.info("Starting document retrieval for training set...")
    document_retrieval_main(db_path, document_k_wiki, raw_training_set,
                            training_doc_file,
                            document_add_claim, document_parallel)
    logger.info("Finished document retrieval for training set.")
    logger.info("Starting document retrieval for dev set...")
    document_retrieval_main(db_path, document_k_wiki, raw_dev_set, dev_doc_file,
                            document_add_claim, document_parallel)
    logger.info("Finished document retrieval for dev set.")
#    logger.info("Starting document retrieval for test set...")
#    document_retrieval_main(db_path, document_k_wiki, raw_test_set, test_doc_file,
#                            document_add_claim, document_parallel)
#    logger.info("Finished document retrieval for test set.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='/path/to/config/file, in JSON format')
    args = parser.parse_args()
    LogHelper.setup()
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    
    logger.info(
            "=========================== Sub-task 1. Document Retrieval ==========================================")
    document_retrieval(logger)

#---