from .Generic import Evaluator
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from transformers import RobertaTokenizer    
from ReinforcementLearning.Custom.Config import QuantizationConfig 
import torch

class QAEvaluator(Evaluator):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
 
    def compute_f1_and_balanced_accuracy(self, ground_truth, predictions):
        possible_labels = super().get_possible_labels()

        gt_tokens = [set(gt.split()) for gt in ground_truth]
        pred_tokens = [set(pred.split()) for pred in predictions]

        tp = sum(len(gt & pred) for gt, pred in zip(gt_tokens, pred_tokens))
        fp = sum(len(pred) - len(gt & pred) for pred, gt in zip(pred_tokens, gt_tokens)) 
        fn = sum(len(gt) - len(gt & pred) for gt, pred in zip(gt_tokens, pred_tokens)) 
        tn = sum(len(possible_labels - (gt | pred)) for gt, pred in zip(gt_tokens, pred_tokens))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (sensitivity + specificity) / 2.0

        return f1, balanced_accuracy
    
    def compute_reciprocal_rank(self, ground_truth, probabilities):
        sorted_labels = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )

        ranked_labels = [label for label, _ in sorted_labels]

        if ground_truth in ranked_labels:
            rank = ranked_labels.index(ground_truth) + 1
            return 1.0 / rank
        return 0.0
    
    def compute_mean_reciprocal_rank(self, all_ground_truth, all_probabilities):
        rrs = []

        for gt, probs in zip(all_ground_truth, all_probabilities):
            rr = self.compute_reciprocal_rank(gt, probs)
            rrs.append(rr)

        return sum(rrs) / len(rrs)
    

    @staticmethod
    def _safe_mean(values):
        cleaned = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                continue
            cleaned.append(float(value))
        if not cleaned:
            return float("nan")
        return float(np.mean(cleaned))
    
    def evaluate(self, ground_truth, predictions, probs):
        exact_matches = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
        em = exact_matches / len(ground_truth) if len(ground_truth) > 0 else 0 

        f1, balanced_accuracy = self.compute_f1_and_balanced_accuracy(ground_truth, predictions)

        mrr = self.compute_mean_reciprocal_rank(ground_truth, probs)


        kl_scores = [
            self.calculate_kl_divergence(prob_dict, actual)
            for prob_dict, actual in zip(probs, ground_truth)
        ]
        kl_avg = self._safe_mean(kl_scores)

        brier_scores = [
            self.multiclass_brier_score(prob_dict, actual)
            for prob_dict, actual in zip(probs, ground_truth)
        ]
        brier_score = self._safe_mean(brier_scores)

        perplexity = np.exp(kl_avg) if np.isfinite(kl_avg) and kl_avg < 700 else float('inf')

        print(f"Evaluation Results for {self.model}:")

        print(f"Exact Match: {em:.4f}")
        print(f"F1 Score {f1:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Mean Reciprocal Rank: {mrr:.4f}")
        print(f"Brier Score: {brier_score:.4f}")
        print(f"KL-Divergence: {kl_avg:.4f}")
        print(f"Perplexity: {perplexity:.4f}")

        print("-" * 50)

        return {
            "EM": em,
            "F1": f1,
            "BA": balanced_accuracy,
            "MRR": mrr,
            "BS": brier_score,
            "KL": kl_avg,
            "Perp": perplexity,
        }


class VisionEvaluator(QAEvaluator):
    def __init__(self, model):
        super().__init__(model)
    
    def evaluate(self, ground_truth, predictions, probs):
        exact_matches = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
        em = exact_matches / len(ground_truth) if len(ground_truth) > 0 else 0

        f1, balanced_accuracy = self.compute_f1_and_balanced_accuracy(ground_truth, predictions)
        mrr = self.compute_mean_reciprocal_rank(ground_truth, probs)

        kl_scores = [
            self.calculate_kl_divergence(prob_dict, actual)
            for prob_dict, actual in zip(probs, ground_truth)
        ]
        kl_avg = self._safe_mean(kl_scores)
        brier_scores = [
            self.multiclass_brier_score(prob_dict, actual)
            for prob_dict, actual in zip(probs, ground_truth)
        ]
        brier_score = self._safe_mean(brier_scores)

        print(f"Evaluation Results for {self.model}:")
        print(f"Exact Match: {em:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Mean Reciprocal Rank: {mrr:.4f}")
        print(f"Brier Score: {brier_score:.4f}")
        print(f"KL-Divergence: {kl_avg:.4f}")

        print("-" * 50)

        return {
            "EM": em,
            "F1": f1,
            "BA": balanced_accuracy,
            "MRR": mrr,
            "BS": brier_score,
            "KL": kl_avg,
        }


def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    if token_ids_1 is None:
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
    return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]


quantisation_config = QuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)


class ReportEvaluator(QAEvaluator):
    def __init__(self, model):
        super().__init__(model)
        self.sentence_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    @staticmethod
    def _token_overlap(gt_text, pred_text):
        gt_tokens = gt_text.split()
        pred_tokens = pred_text.split()

        gt_set = set(gt_tokens)
        pred_set = set(pred_tokens)

        overlap = len(gt_set & pred_set)
        precision = overlap / len(pred_set) if pred_set else 0.0
        recall = overlap / len(gt_set) if gt_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return precision, recall, f1

    def cosine_similarity(self, gt_text, pred_text):
        cos = torch.nn.CosineSimilarity(dim=0)
        gt_embedding = self.sentence_embedding_model.encode(gt_text, convert_to_tensor=True)
        pred_embedding = self.sentence_embedding_model.encode(pred_text, convert_to_tensor=True)
        similarity = cos(gt_embedding, pred_embedding)
        if isinstance(similarity, torch.Tensor):
            return float(similarity.detach().cpu().item())
        return float(similarity)
    
    def batch_cosine_similarity(self, gt_texts, pred_texts):
        sims = []
        for gt, pred in zip(gt_texts, pred_texts):
            sims.append(self.cosine_similarity(gt, pred))
        return self._safe_mean(sims)

    def batch_with_bins_miscalibration(self, gt_texts, pred_texts, confidences, n_bins=10):
        """Calculate Expected Calibration Error using bins."""
        accuracies = []
        for gt, pred in zip(gt_texts, pred_texts):
            sim = (self.cosine_similarity(gt, pred) + 1) / 2
            is_correct = 1 if sim > 0.5 else 0  # or your threshold
            accuracies.append(is_correct)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins, right=False)
        ece = 0.0
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if not np.any(mask):
                continue
            
            avg_conf = np.mean(np.array(confidences)[mask])
            avg_acc = np.mean(np.array(accuracies)[mask])
            bin_weight = np.sum(mask) / len(confidences)
            ece += bin_weight * abs(avg_conf - avg_acc)
        
        return ece
        
    def per_sample_miscalibration(self, gt_text: str, pred_text: str, confidence: float) -> float:
        acc = (self.cosine_similarity(gt_text, pred_text) + 1) / 2

        return abs(confidence - acc)

    def batch_per_sample_miscalibration(self, gt_texts, pred_texts, confidences):
        vals = []
        for gt, pred, conf in zip(gt_texts, pred_texts, confidences):
            ECE = self.per_sample_miscalibration(gt, pred, conf)
            vals.append(ECE)
        return self._safe_mean(vals)

    def perplexity(self, probabilities): 
        probs = [p['probability'] for p in probabilities if p['probability'] > 0]
        if not probs:
            return float('inf')
        log_probs = np.log(probs)
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob) if np.isfinite(avg_log_prob) and avg_log_prob < 700 else float('inf')
        return perplexity

    def evaluate(self, ground_truth, predictions, probs, raw):
        exact_matches = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
        em = exact_matches / len(ground_truth) if len(ground_truth) > 0 else 0

        # BERTScore expects plain Python strings; np.str_ can trigger key lookup errors.
        raw_text = [str(x) for x in raw]
        gt_text = [str(x) for x in ground_truth]
        try:
            RobertaTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
            B_P, B_R, B_F1 = bert_score(
                raw_text,
                gt_text,
                lang="en",
                rescale_with_baseline=True,
                use_fast_tokenizer=True,
            )
            bert_p = float(B_P.mean().item())
            bert_r = float(B_R.mean().item())
            bert_f1 = float(B_F1.mean().item())
        except Exception as error:
            print(f"BERTScore failed for {self.model}: {error}")
            bert_p = float("nan")
            bert_r = float("nan")
            bert_f1 = float("nan")

        token_precisions = []
        token_recalls = []
        token_f1s = []
        phrase_contains = []
        char_similarities = []

        for gt, pred in zip(ground_truth, predictions):
            precision, recall, f1 = self._token_overlap(gt, pred)
            token_precisions.append(precision)
            token_recalls.append(recall)
            token_f1s.append(f1)

            phrase_contains.append(1.0 if gt in pred else 0.0)
            char_similarities.append(SequenceMatcher(None, gt, pred).ratio())

        token_precision = self._safe_mean(token_precisions)
        token_recall = self._safe_mean(token_recalls)
        token_f1 = self._safe_mean(token_f1s)
        contains_rate = self._safe_mean(phrase_contains)
        char_similarity = self._safe_mean(char_similarities)

        cosine_sim = self.batch_cosine_similarity(ground_truth, predictions)

        kl_scores = [
            self.calculate_kl_divergence(prob_dict, actual)
            for prob_dict, actual in zip(probs, ground_truth)
        ]
        kl_avg = self._safe_mean(kl_scores)

        perplexity = self.perplexity(probs)
        confidences = [p['sequence_confidence'] for p in probs]
        miscalib = self.batch_with_bins_miscalibration(ground_truth, predictions, confidences)
       

        reliab = 0
        print(f"Evaluation Results for {self.model}:")
        print(f"Exact Match: {em:.4f}")
        print(f"Token Precision: {token_precision:.4f}")
        print(f"Token Recall: {token_recall:.4f}")
        print(f"Token F1: {token_f1:.4f}")
        print(f"Phrase Containment: {contains_rate:.4f}")
        print(f"Character Similarity: {char_similarity:.4f}")
        print(f"Cosine Similarity: {cosine_sim:.4f}")
        print(f"KL-Divergence: {kl_avg:.4f}")
        print(f"Perplexity: {perplexity:.4f}") 
        print(f"BERTScore P: {bert_p:.4f}")
        print(f"BERTScore R: {bert_r:.4f}")
        print(f"BERTScore F1: {bert_f1:.4f}")
        print(f"Reliability: {reliab:.4f}")
        print(f"Miscalibration: {miscalib:.4f}")


        print("-" * 50)

        return {
            "EM": em,
            "TokP": token_precision,
            "TokR": token_recall,
            "TokF1": token_f1,
            "Contain": contains_rate,
            "CharSim": char_similarity,
            "CosSim": cosine_sim,
            "KL": kl_avg,
            "Perp": perplexity,
            "BERTScore_P": bert_p,
            "BERTScore_R": bert_r,
            "BERTScore_F1": bert_f1,
            "Reliability": reliab,
            "Miscalibration": miscalib
        }
