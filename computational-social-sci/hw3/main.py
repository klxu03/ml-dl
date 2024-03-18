import math
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from tqdm import tqdm

from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)
np.random.seed(42)

import gensim.downloader as api

def get_data(Z_bias=0.05, U_bias=5.0, constant=-0.5):
    """Compile the data from the 20 News Group Scikit-learn distribution
    
    Create a simulated dataset based on a type of news, where the varia-
    ble U is a confounder (whether or not a news document belongs to the
    category or not), Z is the observed variable dependent on U, and Y 
    is the target variable dependent on Z and the confounder.

    Args:
        Z_bias (float, optional): bias of var Z on target. Defaults to 0.05.
        U_bias (float, optional): bias of confounder on target. Defaults to 5.0.
        constant (float, optional): constant bias. Defaults to -0.5.

    Returns:
        np.array: simulated data containing where columns correspond to [Y, Z, U]
        List[str]: list of text in dataset
        vocabulary: vocabulary extracted from text
    """
    newsgroup_data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), shuffle=False)

    target_name_to_label = {
    'comp.graphics': 1,
    'comp.os.ms-windows.misc': 1,
    'comp.sys.ibm.pc.hardware': 1,
    'comp.sys.mac.hardware': 1,
    'comp.windows.x': 1,
    'misc.forsale': 2,
    'rec.autos': 3,
    'rec.motorcycles': 3,
    'rec.sport.baseball': 4,
    'rec.sport.hockey': 4,
    'sci.crypt': 5,
    'sci.electronics': 6,
    'sci.med': 7,
    'sci.space': 8,
    'alt.atheism': 9,
    'soc.religion.christian': 9,
    'talk.religion.misc': 9, 
    'talk.politics.guns': 10,
    'talk.politics.mideast': 10,
    'talk.politics.misc': 10,
    }
    label_names = ["comp", "sale", "auto", "sport", "crypt", "electronics", "med", "space", "religion", "politics"]
    labels = [target_name_to_label[newsgroup_data.target_names[original_label]]
              for original_label in newsgroup_data.target]
    labels = np.array(labels)

    vectorizer = CountVectorizer(min_df=25, max_df=0.01)
    vectorizer.fit(newsgroup_data.data)
    
    dropped_document_idx = set([])
    vocabset = set(vectorizer.get_feature_names_out())
    documents = []
    for idx, text in enumerate(newsgroup_data.data):
        document = []
        for token in text.split(" "):
            if token not in vocabset:
                continue
            document.append(token)

        if len(document) == 0:
            dropped_document_idx.add(idx)
            continue
        documents.append(text)
    labels = np.array([labels[idx]
                    for idx in range(len(labels))
                    if idx not in dropped_document_idx])
    assert(labels.shape[0] == len(documents))
  

    simulated_data = []
    for idx in tqdm(range(len(documents))):
        # confounder
        U = int(labels[idx]==9) # unobserved confounder = religion topic, influences text

        # treatment
        Z = int(1.0*U + np.random.normal(loc=0, scale=1) > 0) # Z depends on U
        Y = constant + Z_bias*Z + U_bias*U + np.random.normal(0.0, 0.1)

        simulated_data.append([Y, Z, U])

    simulated_data = np.array(simulated_data)
    return simulated_data, documents, vectorizer.vocabulary_

def regress_y_on_z(data, maxiter=100):
    """Use the statsmodels library to estimate the treatment effect
    by regression using the sm.OLS model given the variable Z and Y

    Args:
        data (np.array): simulated data
        maxiter: max number of iterations for the regression
    Outputs:
        Use .summary to print results (no return value)
    """

    Y = data[:, 0]
    Z = sm.add_constant(data[:, 1])

    model = sm.OLS(Y, Z)
    res = model.fit(method = "pinv", maxiter=maxiter)

    print(res.summary(yname="Y", xname=["const", "Z"]))
    

def regress_y_on_z_and_u(data, maxiter=2000):
    """Use the statsmodels library to estimate the treatment effect
    by regression using the sm.OLS model given the Z & U and Y

    Args:
        data (np.array): simulated data
        maxiter: max number of iterations for the regression
    Outputs:
        Use .summary to print results (no return value)
    """

    Y = data[:, 0]
    X = sm.add_constant(data[:, 1:3])  

    model = sm.OLS(Y, X)
    res = model.fit(method='pinv', maxiter=maxiter)  

    print(res.summary(yname='Y', xname=['const', 'Z', 'U']))


def regress_y_on_z_and_topics_lda(data, documents, vocabulary, num_topics=50, maxiter=200):
    count_vectorizer = CountVectorizer(max_features=len(vocabulary), stop_words='english', vocabulary=vocabulary)
    dtm = count_vectorizer.fit_transform(documents)

    # Apply LDA to extract topics
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=1, max_iter=maxiter)
    topic_distribution = lda_model.fit_transform(dtm)

    Y = data[:, 0] 
    Z = data[:, 1]  
    X = np.hstack([sm.add_constant(Z.reshape(-1, 1)), topic_distribution])  # Add constant, Z, and topic features
    
    model = sm.OLS(Y, X)
    results = model.fit(method='pinv', maxiter=maxiter)

    column_names = ['const', 'Z'] + [f'Topic_{i}' for i in range(1, num_topics + 1)]

    print(results.summary(xname=column_names))

def regress_y_on_z_and_topics_embedding(data, documents, vocabulary, embedding_type='glove-twitter-50', num_topics=50, maxiter=200):
    print(f"Loading {embedding_type} model. This may take a while the first time...")
    word_vectors = api.load(embedding_type)
    
    # Calculate average embedding for each document
    document_embeddings = np.array([
        np.mean([word_vectors[word] for word in document.lower().split() if word in word_vectors]
                or [np.zeros(word_vectors.vector_size)], axis=0)
        for document in documents
    ])
    
    Y = data[:, 0]  
    Z = data[:, 1]  
    X = np.hstack([sm.add_constant(Z.reshape(-1, 1)), document_embeddings])  # Add constant, Z, and word embeddings
    
    model = sm.OLS(Y, X)
    results = model.fit(method='pinv', maxiter=maxiter)
    
    column_names = ['const', 'Z'] + [f'Feature_{i}' for i in range(1, word_vectors.vector_size + 1)]
    
    print(results.summary(xname=column_names))

def regress_y_on_z_and_topics_tfidf_weighted_embedding(data, documents, vocabulary, embedding_type='glove-twitter-50', num_topics=50, maxiter=200):
    print(f"Loading {embedding_type} model. This may take a while the first time...")
    word_vectors = api.load(embedding_type)
    print(f"Finished loading the model")
    
    # Initialize TF-IDF Vectorizer and transform the documents
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    
    print(f"about to create doc_word_tfidf")
    # Convert TF-IDF matrix to a Dictionary {document_index: {word: tfidf_score}}
    doc_word_tfidf = [{word: tfidf_matrix[idx, feat_idx] 
                       for word, feat_idx in tfidf_vectorizer.vocabulary_.items() 
                       if tfidf_matrix[idx, feat_idx] > 0} 
                      for idx in tqdm(range(tfidf_matrix.shape[0]))]
    
    print("about to start the document embeddings for loop")
    # Calculate weighted average of word vectors for each document
    document_embeddings = np.array([
        np.mean([word_vectors[word] * doc_word_tfidf[doc_idx].get(word, 0) 
                 for word in document.lower().split() if word in word_vectors and word in doc_word_tfidf[doc_idx]] 
                or [np.zeros(word_vectors.vector_size)], axis=0)
        for doc_idx, document in tqdm(enumerate(documents), total=len(documents))
    ])
    
    Y = data[:, 0] 
    Z = data[:, 1] 
    X = np.hstack([sm.add_constant(Z.reshape(-1, 1)), document_embeddings])  # Add constant, Z, and word embeddings
    
    model = sm.OLS(Y, X)
    results = model.fit(method='pinv', maxiter=maxiter)
    
    column_names = ['const', 'Z'] + [f'Feature_{i}' for i in range(1, word_vectors.vector_size + 1)]
    
    print(results.summary(xname=column_names))

def regress_y_on_z_and_topics(data, documents, vocabulary, num_topics=50, maxiter=200):
    """Use a topic model to represent text as a proxy for the unknown confounder
     Then use the statsmodels library to estimate the treatment effect
    by regression using the sm.OLS model given the Z & topics and Y

    Args:
        data (np.array): simulated data
        documents: text data as returned by get_data
        vocabulary: vocabulary as returned by get_data
        num_topics: number of topics for topic model
        maxiter: max number of iterations for the regression
    Outputs:
        Use .summary to print results (no return value)
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=len(vocabulary), stop_words='english', vocabulary=vocabulary)
    tfidf = tfidf_vectorizer.fit_transform(documents)

    nmf_model = NMF(n_components=num_topics, random_state=1)
    topic_distribution = nmf_model.fit_transform(tfidf)

    Y = data[:, 0]  
    Z = data[:, 1] 
    X = np.hstack([sm.add_constant(Z.reshape(-1, 1)), topic_distribution])  # Add constant, Z, and topic features
    
    model = sm.OLS(Y, X)
    results = model.fit(method='pinv', maxiter=maxiter)

    column_names = ['const', 'Z'] + [f'Topic_{i}' for i in range(1, num_topics + 1)]

    print(results.summary(xname=column_names))


# Return the adjusted and unadjusted average treatment effect
def reweigh_with_propensity_scores(data, documents, vocabulary):
    """Given the simulated data, documents, and vocab, reweigh
       the data with inverse probability weighting


    Args:
        data (np.array): simulated data
        documents: text data as returned by get_data
        vocabulary: vocabulary as returned by get_data
    Outputs:
        return [unadjusted ATE], [adjusted ATE]
    """
    X_train_docs, X_test_docs, train_data, test_data = train_test_split(documents, data, test_size=0.5, random_state=1)

    tfidf_vectorizer = TfidfVectorizer(max_features=len(vocabulary), stop_words='english', vocabulary=vocabulary)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_docs)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_docs)

    logistic_regression = LogisticRegression(max_iter=2000, solver='liblinear')
    logistic_regression.fit(X_train_tfidf, train_data[:, 1])  # Z is at index 1

    propensity_scores = logistic_regression.predict_proba(X_test_tfidf)[:, 1]

    weights = np.where(test_data[:, 1] == 1, 1 / propensity_scores, 1 / (1 - propensity_scores))

    # Compute outcomes
    Y = test_data[:, 0]  # Outcome variable at index 0
    Z = test_data[:, 1]  # Treatment variable at index 1

    # Unadjusted ATE
    unadjusted_ate = np.mean(Y[Z == 1]) - np.mean(Y[Z == 0])

    # Adjusted ATE using weights for the treated and control groups
    treated_effect = np.average(Y[Z == 1], weights=weights[Z == 1])
    control_effect = np.average(Y[Z == 0], weights=weights[Z == 0])
    adjusted_ate = treated_effect - control_effect

    return unadjusted_ate, adjusted_ate

if __name__ == "__main__":
    Z_bias = 0.05
    U_bias = 5.0
    constant = -0.5
    data, documents, vocab = get_data(Z_bias=Z_bias, U_bias=U_bias, constant=constant)
    print(f"DATA: Y = {constant} + {Z_bias}*Z + {U_bias}*U")
    
    print("\nEstimating the treatment effect by regressing Y on Z only\n")
    regress_y_on_z(data)
    
    print("\n*******************************************************************************************")
    print("Estimating the treatment effect by regressing Y on Z and the confounder U\n")
    regress_y_on_z_and_u(data)
    
    print("\n*******************************************************************************************")
    print("Estimating the treatment effect by regressing Y on Z and the structured text\n")
    regress_y_on_z_and_topics(data, documents=documents, vocabulary=vocab, num_topics=50)

    print("\n*******************************************************************************************")
    print("Estimating the treatment effect by weighting with propensity scores\n")
    unadjusted, adjusted = reweigh_with_propensity_scores(data, documents=documents, vocabulary=vocab)
    print("Adjusted", adjusted)
    print("Unadjusted", unadjusted)

    EMBEDDING_TYPES = ["glove-twitter-50", "glove-twitter-200", "word2vec-google-news-300"]

    for embedding in EMBEDDING_TYPES:
        print("\n*******************************************************************************************")
        print(f"Embedding Type: {embedding}")

        print("\n*******************************************************************************************")
        print("Estimating the treatment effect by regressing Y on Z and the structured text with Word2Vec Embeddings\n")
        regress_y_on_z_and_topics_embedding(data, documents=documents, embedding_type=embedding, vocabulary=vocab, num_topics=20)

        # print("\n*******************************************************************************************")
        # print("Estimating the treatment effect by regressing Y on Z and the structured text with TF-IDF Weighted Word2Vec Embeddings\n")
        # regress_y_on_z_and_topics_tfidf_weighted_embedding(data, documents=documents, embedding_type=EMBEDDING_TYPES[1], vocabulary=vocab, num_topics=20)

    print("\n*******************************************************************************************")
    print("Estimating the treatment effect by regressing Y on Z and the structured text with LDA Topics\n")
    regress_y_on_z_and_topics_lda(data, documents=documents, vocabulary=vocab, num_topics=50)
