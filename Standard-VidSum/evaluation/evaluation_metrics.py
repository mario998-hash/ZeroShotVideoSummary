import numpy as np
import csv

def evaluate_summary_fscore(predicted_summary, user_summary, eval_method):
    ''' 
    Function that evaluates the predicted summary using F-Score. 

    Inputs:
        predicted_summary: numpy (binary) array of shape (n_frames)
        user_summary: numpy (binary) array of shape (n_users, n_frames)
        eval_method: method for combining the F-Scores for each comparison with user summaries - values: 'avg' or 'max'
    Outputs:
        max (TVSum) or average (SumMe) F-Score between users 

    '''
    max_len = max(len(predicted_summary),user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        # Compute precision, recall, f-score
        precision = 0 if sum(S) == 0 else sum(overlapped)/sum(S)
        recall = 0 if sum(G) == 0 else sum(overlapped)/sum(G)
        if (precision+recall==0):
            f_scores.append(0)
        else:
            f_scores.append(2*precision*recall*100/(precision+recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)


def evaluate_summary_fscore_all(predicted_summary, user_summary, eval_method):
    ''' 
    Function that evaluates the predicted summary using F-Score. 

    Inputs:
        predicted_summary: numpy (binary) array of shape (n_frames)
        user_summary: numpy (binary) array of shape (n_users, n_frames)
        eval_method: method for combining the F-Scores for each comparison with user summaries - values: 'avg' or 'max'
    Outputs:
        max (TVSum) or average (SumMe) F-Score between users 

    '''
    max_len = max(len(predicted_summary),user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    results = []
    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        # Compute precision, recall, f-score
        precision = 0 if sum(S) == 0 else sum(overlapped)/sum(S)
        recall = 0 if sum(G) == 0 else sum(overlapped)/sum(G)
        if (precision+recall==0):
            f_scores.append(0)
            results.append((0,0,0))
        else:
            f_scores.append(2*precision*recall*100/(precision+recall))
            results.append(
                [
                    precision * 100,
                    recall * 100,
                    (2 * precision * recall * 100 / (precision + recall)),
                ]
            )
    results = np.array(results)
    
    if eval_method == "max":
        x = [y[2] for y in results]
        max_values = np.argmax(x)
        return results[max_values]
    else:
        return np.mean(results, axis=0)  
