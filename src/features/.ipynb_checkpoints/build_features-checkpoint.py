import random
from tqdm import tqdm
import re
from itertools import chain
from collections import Counter
import spacy
import numpy as np
from sklearn.metrics import classification_report
from spacy import displacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.util import filter_spans
nlp = spacy.load("en_core_web_lg")

import sys
sys.path.insert(0, '/src/models/')
from predict_model import SpanPredictor as classify

def random_text_splitter(text):
    
    """
    Random breaks up a text into an X amount of sentences. 
    The output sentences consist of a minimum of 10 sentences.
    """

    # Split text
    words = text.split()
    # Get the amount of words
    word_amount = len(words)
    # Create counter
    remaining_word_amount = word_amount
    # Init list
    parts = []
    # While words remaining
    while remaining_word_amount > 0:
        if len(words) < 10:
            # Add last part if less then 10
            parts[-1] = parts[-1] + ' '.join(words)
            # exit
            remaining_word_amount = 0
        # Generate random int
        randint = random.randint(10, word_amount)
        # Append to list 
        parts.append(' '.join(words[:randint]))
        # Delete previous selection
        words = words[randint:]
        # Update counter
        remaining_word_amount -= randint
        
    return parts

def text_cleaner(dirty_text, per_sent=True):
    
    """
    Cleans the contents of a string object and uses SpaCy to return single sentences.
    """    
    
    regexes = [
        (r'\(\d+.+?Close\n\t\n\)', ''),
        (r'\(.+?\)', ''),
        (r'\[.+?\]', ''),
        (r'cm\.', 'centimeters'),
        (r'm\.', 'meters'),
        (r'ft\.', 'feet'),
        (r'\.\.\.', '.'),
        (r'\.\s*\.', '.'),
        (r'-*subsp\.', 'subspecies'),
        (r'-*var\.', 'variation'),
    ]

    # Clean text
    for regex, replace in regexes:
        dirty_text = re.sub(regex, replace, dirty_text)
    # Clean stuff
    text = dirty_text.replace('\r', "")\
                 .replace('\n', "")\
                 .replace('\t', "")\
                 .strip()
                 #.encode("ascii", "ignore")\
                 #.decode()\
    
    #nlp
    doc = nlp(text)
    sents = [i for i in doc.sents]
    
    sents_clean = []
    # Clean non English
    for sentence in sents:
        # Skip short stuff
        if len(sentence) <= 5:
            continue
        # Create ratio
        non_eng = [token.is_oov for token in sentence].count(True)
        # Continue if the ratio is bad (non English jibberisch)
        if non_eng != 0:
            if non_eng / len(doc) > .2:
                continue
        sents_clean.append(sentence.text)
    
    sents_clean = list(set(sents_clean))
    
    if per_sent:
        return sents_clean
    else:
        return doc
    

def get_prediction_results(dictionary, model,
                           index=-1, 
                           soft_error=False, 
                           beta=0.95):
    
    """
    Uses a dictionary with species names. Undicts the dict and returns
    a precision/recall plot that can be printed. The second value returned
    contains a list with missclassified sentences. Optionally a soft_error can 
    used.
    """

    # Get dict values
    data_values = (list(chain.from_iterable(dictionary.values())))
    # init arrays and list
    y_list = np.array([])
    pred_list = np.array([])
    misclass_list = []
    
    # loop over the values of the list
    for (label, span) in tqdm(data_values[0:index]):
        # Clean the sentence
        sentences = text_cleaner(span)
        # Loop over the sentences
        for sent in sentences:
            sent_str = sent.text
            
            if soft_error:
                pred = classify(sent_str, model=model, pred_values=True)
                pred_np = pred[1][1].numpy().item()
                prediction = pred[0]
                if beta < pred_np or pred_np < 1-beta:
                    label = pred[0]
            else:
                # Get prediciton
                prediction = classify(sent_str, model=model)
            # Stack horizontally
            pred_list = np.hstack([pred_list, prediction])
            y_list = np.hstack([y_list, label])
            # Append the missclassified sents
            if label != prediction:
                misclass_list.append(tuple([f'real:{label} pred:{prediction}', sent_str]))
    
    # Generate pres/recall            
    report = classification_report(y_list, pred_list) 
    
    return report, misclass_list