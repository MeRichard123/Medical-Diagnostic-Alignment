from abc import ABC, abstractmethod
import numpy as np
from utils import reference_impl
import pandas as pd

class Evaluator(ABC):
    BASE_DIR = "./"
    def __init__(self, model):
        self.DATA_MODEL = model

    def get_possible_labels(self):
        data = pd.read_csv("./data/processed_iuxray_mcqa_dataset.csv")
        labels = set()
        for col in ['opt1', 'opt2', 'opt3', 'opt4']:
            labels.update(data[col].unique())
        return labels
    
    @reference_impl
    def calculate_kl_full(self, predicted_probs, actual_outcome):
        classes = list(predicted_probs.keys())

        # Construct one-hot true distribution P
        P = np.array([1.0 if c == actual_outcome else 0.0 for c in classes])

        # Predicted distribution Q
        Q = np.array([predicted_probs[c] for c in classes])
        Q = np.clip(Q, 1e-12, 1.0)

        return np.sum(P * np.log(P / Q + 1e-12))    

    """
    For a single sample, the correct KL divergence for multiclass classification is:
    D_KL(P∥Q) = ∑_c P(c) log⁡P(c) / Q(c) 
    
    But because  P(c∗)=1  for the true class and 0 for all others:

    D_KL = −log⁡ Q(c∗)

    So evaluating KL reduces to negative log likelihood, using the probability assigned to the true class.
    """
    
    def calculate_kl_divergence(self, predicted_probs, actual_outcome):
        if actual_outcome == 'Error':
            return float('-inf')  # or some large number to indicate error
        if isinstance(predicted_probs, dict) and 'type' not in predicted_probs:
            q = predicted_probs[actual_outcome]     # predicted probability for true class
            q = max(q, 1e-10)                        # avoid log(0)

            return -np.log(q)
        else:
            gt_logprob = predicted_probs.get('ground_truth_logprob')
            return -gt_logprob
    
    def multiclass_brier_score(self, prob_dict, actual_class):
        # Convert dict to arrays
        if isinstance(prob_dict, dict) and 'type' not in prob_dict:
            classes = list(prob_dict.keys())
            y_prob = np.array([prob_dict[c] for c in classes])

            y_true = np.array([1 if c == actual_class else 0 for c in classes])

            return np.sum((y_prob - y_true)**2)
        else:
            return None # no Brier for Open Ended Tasks
        
    @abstractmethod
    def evaluate(self, ground_truth, predictions):
        return

    

