import joblib, os
import gensim
import pandas as pd
import datetime


def omit_senate_special_language(x, language):
    captured = [token for token in language if x.find(token) >= 0]
    return False if captured else True

def remove_phrases(speech,phrases):
    for phrase in phrases:
        speech.replace(phrase,'')
    return speech

def merge_texts(corpus_list):
    Texts = []
    for corpus in corpus_list:
        Texts.extend(pd.read_csv(corpus)['speech_processed'])
    Texts = [text.split() for text in Texts]
    return Texts
