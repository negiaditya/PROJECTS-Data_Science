import pandas as pd
import numpy as np
from flair.data import Sentence
from flair.models import SequenceTagger
import logging
import json
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BytePairEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import re
from product_config import colors_list,size_list,size_word

def get_model(trained_model_path):
	print("Loading pretrained model for brand prediction.")
	return SequenceTagger.load(trained_model_path)

def extract_color(x,colors=colors_list):
    print("Getting possible colors in product title.")
    col_lst = []
    for color in colors:
        if color[::-1] in str(x).lower()[::-1]:
            col_lst.append(color) 
    return col_lst

def extract_brand(x,model):
    print("Predicting brand in product title.")
    sen = Sentence(str(x))
    model.predict(sen)
    if re.findall(r"\[.*?]", sen.to_tagged_string()):
        return re.findall(r'\[.*?]', sen.to_tagged_string())[-1].split('/BRAND')[0][2:-1]
    else:
        return None

def extract_size(x,size_lst = size_list,size_wrd = size_word):
    print("Getting possible sizes in product title.")
    fin_lst = []
    str_lst = str(x).split(' ')
    for st in str_lst:
        if re.findall(r"\d*\.?\d+",st):
            fin_lst.append(st)
        if st in size_lst:
            fin_lst.append(st)
    for wrd in size_wrd:
        if wrd in str(x).upper():
            fin_lst.append(wrd)
    return fin_lst

def get_attributes(text,model):
	color = extract_color(text)
	print("Done with color tagging.")
	size = extract_size(text)
	print("Done with size tagging.")
	brand = extract_brand(text,model)
	print("Done with brand tagging.")
	return {"color":color,"size":size,"brand":brand}

if __name__ == '__main__':
	input_text = "Helga Sandal Dark Brown Leather 7 Medium"
	trained_model_path = "artifacts/best-model.pt"
	model = get_model(trained_model_path)
	print(get_attributes(input_text,model))

