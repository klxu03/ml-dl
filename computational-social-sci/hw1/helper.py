import pandas
from collections import Counter, defaultdict
import math

def read_data():
    """
    Reads data from the data/ folder.

    Returns:
        pd.DataFrame: dataframe containing the text and metadata
    """
    data = open("data/cr_111-114.txt").readlines()
    meta_df = pandas.read_csv("data/cr_metadata.csv")
    
    ### verifying the length of text file and metadata file are the same
    assert(len(data) == len(meta_df))
    
    ### merging files
    meta_df["text"] = data
    
    return meta_df

politics_words = [
                  'freedom', 'justice', 'equality', 'democracy', # political abstractions
                  'abortion', 'immigration', 'welfare', 'taxes', # partisan political issues   
                  'democrat', 'republican' # political parties               
                 ] # from Rodriguez and Spirling 2021

