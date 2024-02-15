import pandas
from collections import Counter, defaultdict
import math
from tabulate import tabulate
import copy

from helper import read_data, politics_words


def compute_log_odds_ratios(a_counter, b_counter):
    """ Calculate the log odds ratio given the counters for group A and
    group B.

    Parameters:
    ===========
    a_counter (collections.Counter): word counts for group A
    b_counter (collections.Counter): word counts for group B

    Returns:
    ========
    log_odds_ratio (dict): dictionary containing the log odds ratio of each
    word {"word" (str): ratio (float)}
    """ 
    log_odds_ratio = {}
    ####### PART A #######
    total_a = sum(a_counter.values())
    total_b = sum(b_counter.values())

    for word in set(a_counter).intersection(b_counter):
        # Get the word counts for each group, default to 1 if not found
        a_word_count = a_counter.get(word, 0)
        b_word_count = b_counter.get(word, 0)
        
        # Calculate the normalized frequencies
        f_a = a_word_count / total_a
        f_b = b_word_count / total_b
        
        # Calculate the odds
        odds_a = f_a / (1 - f_a)
        odds_b = f_b / (1 - f_b)
        
        # Calculate the log odds
        log_odds_a = math.log(odds_a)
        log_odds_b = math.log(odds_b)
        
        # Calculate the log odds ratio and store it in the dictionary
        log_odds_ratio[word] = log_odds_a - log_odds_b
    
    return log_odds_ratio


def compute_odds_with_prior(counts1, counts2, prior):
    """ Calculate the log odds ratio with a prior given the counters
    for group A, group B and the prior.

    Parameters:
    ===========
    a_counter (collections.Counter): word counts for group A
    b_counter (collections.Counter): word counts for group B
    prior     (collections.Counter): word counts for the prior
    
    Returns:
    ========
    log_odds_ratio (dict): dictionary containing the log odds ratio of each
    word {"word" (str): ratio (float)}
    """ 
    log_odds_ratio = {}

    ####### PART B #######
    total_counts1 = sum(counts1.values())
    total_counts2 = sum(counts2.values())

    alpha_0 = sum(prior.values())

    for word in set(counts1).intersection(counts2):
        y_w1 = counts1.get(word, 0) 
        y_w2 = counts2.get(word, 0) 
        alpha_w = prior.get(word, 0)
        
        omega_w1 = (y_w1 + alpha_w) / (total_counts1 + alpha_0 - y_w1 - alpha_w)
        omega_w2 = (y_w2 + alpha_w) / (total_counts2 + alpha_0 - y_w2 - alpha_w)
        
        delta_w = math.log(omega_w1 / omega_w2)
        
        variance = 1 / (y_w1 + alpha_w) + 1 / (y_w2 + alpha_w)
        z_score = delta_w / math.sqrt(variance)
        
        log_odds_ratio[word] = z_score

    return log_odds_ratio

def smoother(A, window, count_list):
    """ smooth the counts given a window and a smoothing factor A.

    Parameters:
    ===========
    A            (float): smoothing factor
    window       (int): window length of the moving average
    count_list   (List[collections.Counter]): a list of counters to smooth
    
    Returns:
    ========
    new_counts    (List[collections.Counter]): a smoothed list of counters
    """ 
    new_counts = []
    
    ####### PART C #######

    # Initialize the moving count with the first `window` elements
    moving_count = Counter()
    for i in range(window):
        moving_count.update(count_list[i])

    # Start applying exponential smoothing after the first `window` elements
    for t in range(window, len(count_list)):
        current_counter = count_list[t]
        previous_smoothed_counter = new_counts[-1] if len(new_counts) > 0 else Counter()

        # Update the moving count
        # Subtract counts going out of the window
        if t > window:
            moving_count.subtract(count_list[t-window])
        # Add counts coming into the window
        moving_count.update(count_list[t])

        # Calculate the smoothed value for the current time point
        smoothed_counter = Counter()
        for word in current_counter:
            m_wt = moving_count[word]
            s_wt_previous = previous_smoothed_counter[word]
            s_wt = A * m_wt + (1 - A) * s_wt_previous
            smoothed_counter[word] = s_wt

        new_counts.append(smoothed_counter)
    
    return new_counts
            


if __name__ == "__main__":
    data_df = read_data()
    
    # Separate data
    r_df = data_df[data_df.party == "R"]
    d_df = data_df[data_df.party == "D"]

    r_m_df = data_df[(data_df.party == "R") & (data_df.gender == "M")]
    d_f_df = data_df[(data_df.party == "D") & (data_df.gender == "F")]

    r_counter = Counter()
    r_df.text.apply(lambda t: r_counter.update(t.split()))

    r_m_counter = Counter()
    r_m_df.text.apply(lambda t: r_m_counter.update(t.split()))

    d_counter = Counter()
    d_df.text.apply(lambda t: d_counter.update(t.split()))

    d_f_counter = Counter()
    d_f_df.text.apply(lambda t: d_f_counter.update(t.split()))
    
    # Part A. Compute Log-Odds Ratio
    w_to_ratio = compute_log_odds_ratios(d_counter, r_counter)
    print("More Republican")
    table = []
    for w, odds in sorted(w_to_ratio.items(), key=lambda item: item[1])[:10]:
        table.append([w, odds, r_counter[w], d_counter[w]])
    print(tabulate(table, headers=["Word", "Odds", "R count", "D count"]))

    print()
    print("More Democrat")
    table = []
    for w, odds in sorted(w_to_ratio.items(), key=lambda item: item[1], reverse=True)[:10]:
        table.append([w, odds, r_counter[w], d_counter[w]])
    print(tabulate(table, headers=["Word", "Odds", "R count", "D count"]))
    
    # Part B. Compute Log-Odds Ratio with Prior
    prior = copy.deepcopy(d_counter)
    prior.update(r_counter)

    w_to_ratio = compute_odds_with_prior(d_f_counter, r_m_counter, prior)
    print()
    print("More Republican")
    table = []
    for w, odds in sorted(w_to_ratio.items(), key=lambda item: item[1])[:10]:
        table.append([w, odds, r_m_counter[w], d_f_counter[w]])
    print(tabulate(table, headers=["Word", "Odds", "R_M count", "D_F count"]))


    print()
    print("More Democrat")
    table = []
    for w, odds in sorted(w_to_ratio.items(), key=lambda item: item[1], reverse=True)[:10]:
       table.append([w, odds, r_m_counter[w], d_f_counter[w]])
    print(tabulate(table, headers=["Word", "Odds", "R_M count", "D_F count"]))

        
    # Part C. Compute word evolutions
    d_counters = []
    r_counters = []
    
    for i, session_id in enumerate(data_df.session_id.unique()):
        r_df = data_df[(data_df.party == "R")&(data_df.session_id == session_id)]
        d_df = data_df[(data_df.party == "D")&(data_df.session_id == session_id)]
        r_counter = Counter()
        r_df.text.apply(lambda t: r_counter.update(t.split()))
        r_counters.append(r_counter)

        d_counter = Counter()
        d_df.text.apply(lambda t: d_counter.update(t.split()))
        d_counters.append(d_counter)
    new_d_counters = smoother(A=.2, window=1, count_list=d_counters)
    new_r_counters = smoother(A=.2, window=1, count_list=r_counters)
    
    print()
    print("Changes over time in log odds with prior")
    political_keywords = {w: [] for w in politics_words}
    for i, (d_counter, r_counter) in enumerate(zip(new_d_counters, new_r_counters)):
        w_to_ratio = compute_odds_with_prior(d_counter, r_counter, prior)
        for w, odds_list in political_keywords.items():
            political_keywords[w].append(w_to_ratio[w])
    table = []
    for w, odds_list in political_keywords.items():
        table.append([w] + odds_list)
    print(tabulate(table, headers=["Word", "112", "113", "114"]))