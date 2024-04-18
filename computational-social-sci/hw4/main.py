import re
import pandas as pd
import argparse
import spacy
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import numpy as np
import scipy
import gc
import seaborn as sns 
from matplotlib import pyplot as plt

nlp = spacy.load("en_core_web_sm")

def read_data():
    return pd.read_csv("acl_anthology_abstracts_llm.csv")


def get_human_nonhuman_scores(sentence, human, nonhuman, model, tokenizer, device):
    human_inds = [tokenizer.get_vocab()[x] for x in human]
    nonhuman_inds = [tokenizer.get_vocab()[x] for x in nonhuman]
    
    ########################################
    ########### PART 1 #####################
    ########################################
    
    
    
    return human_scores, nonhuman_scores


def get_anthroscore(text, entities, model, tokenizer, device):
    # Mask sentences
    pattern_list = ['\\b%s\\b'%s for s in entities] # add boundaries
    masked_sents = []
    if text.strip():
        doc = nlp(text)
        for _parsed_sentence in doc.sents:
            for _noun_chunk in _parsed_sentence.noun_chunks:
                if _noun_chunk.root.dep_ == 'nsubj' or _noun_chunk.root.dep_ == 'dobj':
                    for _pattern in pattern_list:
                        if re.findall(_pattern.lower(), _noun_chunk.text.lower()):
                                _verb = _noun_chunk.root.head.lemma_.lower()
                                target = str(_parsed_sentence).replace(str(_noun_chunk),'<mask>')
                                masked_sents.append(target)

    if len(masked_sents)==0:
        print("Stopping calculation, no words found.")
        return np.nan
        
    # Get scores
    hterms = ['he', 'she', 'her', 'him', 'He', 'She', 'Her']
    nterms = ['it', 'its', 'It', 'Its']
    anthroscore = 0
    ########################################
    ########### PART 1 #####################
    ########################################
    
    
    
    
    return anthroscore


def main():
    ###### SETUP ############################
    dataset = read_data()
    
    with open("LM_terms.txt") as f:
        LLM_entities = [line.rstrip('\n') for line in f]
    
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("BERT model loaded on %s"%device)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    
    # GETTING ANTHROSCORE WITH DEFAULT TERMS
    dataset['anthroscore'] = dataset.abstract.apply(lambda a: get_anthroscore(a, entities=LLM_entities, model=model, tokenizer=tokenizer, device=device))
    
    # SAVE THIS IMAGE FOR PART 2
    plt.figure(figsize=(15,8))
    ax = sns.lineplot(data=dataset[dataset.year > 2007], x="year", y="anthroscore", errorbar=("ci", 95), err_style="band")
    sns.regplot(data=dataset[dataset.year > 2007], x="year", y="anthroscore", scatter=False, ax=ax, ci=False, color="gray", line_kws={"linestyle":"dashed"})


if __name__ == '__main__':
    main()