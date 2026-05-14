import pandas as pd
import cv2
import numpy as np
import re, os
import pathlib

def normalize_mesh_field(x):
    if pd.isna(x):
        return []
    terms = str(x).split(';')
    return [t.split('/', 1)[0].strip() for t in terms]


def preprocess_image(filename):
    if "aug" in filename:
        img = cv2.imread('./images/augmented/' + filename, cv2.IMREAD_GRAYSCALE)
    else:   
        img = cv2.imread('./images/images_normalized/' + filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img


def train_test_split(df, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return df.iloc[train_indices], df.iloc[test_indices]

def build_mcqa_prompt(df_row):
    question = f"""
    Respond with the most likely diagnosis based on the following findings:

    Options:
    A. {df_row['opt1']}
    B. {df_row['opt2']}
    C. {df_row['opt3']}
    D. {df_row['opt4']}

    --- 
    **Background Information:**
    Based ONLY on the following context, answer the question strictly with a single letter: 'A', 'B', 'C' or 'D'.
    {df_row['findings']}
    

    --- 
    Answer:

    """
    return question, get_correct_option(df_row['copt'], df_row)


def build_vision_prompt(df_row):
    question = f"""
    Respond with the most likely diagnosis based on the following image:

    Options:
    A. {df_row['opt1']}
    B. {df_row['opt2']}
    C. {df_row['opt3']}
    D. {df_row['opt4']}

    --- 
    Answer:

    """
    return question, get_correct_option(df_row['copt'], df_row)


def get_correct_option(copt, df_row):
    if copt == df_row['opt1']:
        return 'A'
    elif copt == df_row['opt2']:
        return 'B'
    elif copt == df_row['opt3']:
        return 'C'
    elif copt == df_row['opt4']:
        return 'D'
    else:
        print(f"Warning: copt '{copt}' does not match any option for row with findings: {df_row['copt']}")

        return 'Error'
    

class ReferenceImplementation(Exception):
    def __init__(self, message):
        super().__init__(message)


def reference_impl(func):
    def inner(*vaargs):
        raise ReferenceImplementation("This is is a reference implemetation serving as documentation for a function \n that was defined in a different way. This should not be called.")
    return inner


def build_label_token_ids(tokeniser, labels):
	label_token_ids = {}
	for label in labels:
		variants = [label, f" {label}", f"\n{label}"]
		token_ids = set()
		for text in variants:
			ids = tokeniser.encode(text, add_special_tokens=False)
			if len(ids) == 1:
				token_ids.add(ids[0])
		label_token_ids[label] = sorted(token_ids)
	return label_token_ids

def json_serialiser(obj):
	if isinstance(obj, (np.integer,)):
		return int(obj)
	if isinstance(obj, (np.floating,)):
		return float(obj)
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def extract_label(text):
	if not text:
		return ""
	match = re.search(r"\b([ABCD])\b", text.upper())
	return match.group(1) if match else ""

def clean_predictions(predicted_label, probabilities, LABELS):
	probs_clean = {label: float(probabilities.get(label, 0.0)) for label in LABELS}
	pred_label = predicted_label if predicted_label in LABELS else max(probs_clean, key=probs_clean.get)
	return pred_label, probs_clean

def resolve_image_path(filename):
	if pd.isna(filename) or str(filename).strip() in ("", "None", "nan"):
		return None

	name = str(filename).strip()
	candidates = [
		pathlib.Path("./data", "images", "processed", name),
	]
	for path in candidates:
		if os.path.exists(path):
			return str(path)
	return None


def build_gen_prompt(df_row):
    question = f"""
    Respond with the most likely diagnosis based on the following findings:
    --- 
    **Background Information:**
    Based ONLY on the following context, answer the question with 1-4 words max describing the diagnosis or "normal" if there is no diagnosis. 

    {df_row['findings']}

    --- 
    Answer:

    """
    return question, df_row['copt']
