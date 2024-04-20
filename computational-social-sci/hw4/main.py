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
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.load("en_core_web_sm")

def read_data():
    return pd.read_csv("custom.csv")
    # return pd.read_csv("acl_anthology_abstracts_llm.csv")


def get_human_nonhuman_scores(sentence, human, nonhuman, model, tokenizer, device):
    human_inds = [tokenizer.get_vocab()[x] for x in human]
    nonhuman_inds = [tokenizer.get_vocab()[x] for x in nonhuman]
    
    ########################################
    ########### PART 1 #####################
    ########################################
    token_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)  # Tokenize the input sentence and convert to tensor
    masked_index = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero() # Find the index of the masked token
    masked_pos = [mask.item() for mask in masked_index][0]

    with torch.no_grad():  # Ensure no gradient is calculated to speed up processing and reduce memory usage
            output = model(token_ids)  # Pass the tokenized sentence through the model

    last_hidden_state = output[0].squeeze()
    mask_hidden_state = last_hidden_state[masked_pos]

    probs = torch.nn.functional.softmax(mask_hidden_state, dim=0)  # Apply softmax to get probabilities

    # Calculate scores by summing probabilities of specific tokens
    human_scores = probs[human_inds].sum(axis=0).item()  # Sum probabilities of human terms
    nonhuman_scores = probs[nonhuman_inds].sum(axis=0).item()  # Sum probabilities of non-human terms
    
    
    
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
    scores = []

    for sent in masked_sents:
        human_scores, nonhuman_scores = get_human_nonhuman_scores(sent, hterms, nterms, model, tokenizer, device)
        if human_scores + nonhuman_scores > 0:  # Avoid division by zero
            score = np.log(human_scores / nonhuman_scores)
            scores.append(score)

    if scores:
        anthroscore = np.mean(scores) # Return the average score
    else:
        return np.nan  # Return NaN if no scores to average
    
    
    
    
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
    dataset['anthroscore'] = dataset.abstract.progress_apply(
        lambda a: get_anthroscore(a, entities=LLM_entities, model=model, tokenizer=tokenizer, device=device)
    )
    dataset.to_csv("anthroscores_custom.csv", index=False)

    
    # SAVE THIS IMAGE FOR PART 2
    plt.figure(figsize=(15,8))
    ax = sns.lineplot(data=dataset[dataset.year > 1989], x="year", y="anthroscore", errorbar=("ci", 95), err_style="band")
    sns.regplot(data=dataset[dataset.year > 2007], x="year", y="anthroscore", scatter=False, ax=ax, ci=False, color="gray", line_kws={"linestyle":"dashed"})

    # Save figure
    plt.savefig('anthroscore_custom.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()