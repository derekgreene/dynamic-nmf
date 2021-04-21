import os, re, joblib,json
import pandas as pd
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import spacy
nlp = spacy.load("en_core_web_sm")

from string import punctuation
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from text.congressional_utils.utils import *
from optparse import OptionParser
from gensim.models.phrases import Phrases


def custom_tokenizer(nlp):
    """
    custom spacy tokenizer for maintaining hyphenated words
    """
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

nlp.tokenizer = custom_tokenizer(nlp)


def tokenize(speech):
    """
    speech tokenizer, lemmatizes only NOUN, PROPN, VERB, ADJ
    """
    # Tokenize and lemmatize
    text = []
    for token in nlp(speech):
        if token.pos_ in ['NOUN','PROPN','VERB','ADJ']:
            text.append(token.lemma_.lower().replace('.',''))
    return text



class speech_processor():
    """
    class for generating and processing speeches
    """
    def __init__(self,path,chamber,omit_tokens):
        self.chamber = chamber
        self.path = path
        self.speeches = None
        self.df = None

        with open(omit_tokens,'r') as _file:
            self.omit_tokens = json.load(_file)

    def generate_speeches_df(self,wc=50,start_date=None,end_date=None,testing=False):
        """
        parse speeches, link to meta-data, and filter

        args:
            - wc: minimum speech length
            - start_date: first date in range
            - end_date: last date in range
        """

        # load in speeches
        speeches = open(os.path.join(self.path,f"speeches_{self.chamber}.txt"),
                        encoding='utf-8',
                        errors='ignore').read().split('\n')
        speeches = {row[:10]:row[11:].strip() for row in speeches}

        # get description file processed
        congress = open(os.path.join(self.path,f"descr_{self.chamber}.txt"),
                         encoding='utf-8',
                         errors='ignore').read()
        rows = [r.split("|") for r in congress.split('\n')[1:-1]]
        columns = congress.split('\n')[0].split('|')
        df = pd.DataFrame(rows, columns=columns)

        # link data
        df['speech_text'] = df.speech_id.apply(lambda x: speeches[x])
        df = df.groupby('speech_text').first().reset_index() # duplicates exist
        df['date'] = pd.to_datetime(df.date)

        # subset by date range
        if start_date:
            df = df.loc[df.date >= start_date]
        if end_date:
            df = df.loc[df.date < end_date]

        # filter data
        self.df = df.loc[(df.gender != "Special") &
                        (df.gender != 'Unknown') &
                        (df.word_count.astype(int) >= wc) &
                        (df.speech_text.apply(omit_senate_special_language,
                                              language = self.omit_tokens['phrases']))]

        if testing:
            print('testing: only using 200 speeches')
            self.df = self.df.sample(200)

    def process_speeches(self, omit_keys = None, min_df = 50, threshold = 10):
        """
        pre-process speeches. Performs normalization, phrase removal, tokenization,
        and bigram collocation.

        args:
            - omit_keys: keys for lists in self.omit_tokens to include in
                tokens to remove
            - min_df: minimum number of documents for collocation
            - threshold: collocation threshold

        """
        # keys to omit
        omit = self.omit_tokens['glossary_terms']

        # normalize text
        text_normalized = self.df['speech_text'].str.lower().str.translate(punctuation)
        # remove phrases
        text_phrased = [remove_phrases(speech,omit) for speech in tqdm(text_normalized,desc="omitting phrases")]
        # tokenize
        text_tokens = [tokenize(speech) for speech in tqdm(text_phrased,desc='tokenizing')]
        # collocation
        bigrams = Phrases(text_tokens, min_count=min_df, threshold=threshold)
        speech_ngrams = [bigrams[sent] for sent in text_tokens]
        # rejoin
        self.df['speech_processed'] = [' '.join(speech) for speech in speech_ngrams]


    def make_dtm(self,min_df=100,max_df=0.3):
        """
        makes a document term matrix for speeches

        args:
            - min_df: sklearn TfIDF min_df arg
            - max_df: sklearn TfIDF max_df arg
        """
        tfidf = TfidfVectorizer(min_df = min_df, max_df = max_df)

        dtm = tfidf.fit_transform(self.df['speech_processed'])

        self.dtm_dict = {"vectorizer":tfidf,
                         "dtm":dtm,
                         "vocab":tfidf.get_feature_names(),
                         "speech_id":self.df['speech_id']}


    def save_speeches_df(self,out_path,filename=None):
        """
        save speeches pandas DataFrame

        args:
            - out_path: directory to save data
            - filname: optional filename

        """
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        if not filename:
            filename = f"{self.chamber}.csv"
        else:
            if not filename.endswith('.csv'):
                raise Exception("File must be csv")

        self.df.to_csv(os.path.join(out_path,filename))


    def save_speeches_dtm(self,out_path,filename=None):
        """
        save speeches DTM and other info

        args:
            - out_path: directory to save data
            - filname: optional filename

        """
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        if not filename:
            filename = f'{self.chamber}.pkl'
        else:
            if not filename.endswith('.pkl'):
                raise Exception("File must be pkl")

        joblib.dump(self.dtm_dict,os.path.join(out_path,filename))



def main(path, chamber, omit_tokens, out_path, wc=50, start_date=None, end_date = None,
         ngram_min_df = 50, ngram_thresh=10, dtm_min_df = 30,
         dtm_max_df = 0.3, filename=None,testing=False):

    processor = speech_processor(path,chamber,omit_tokens)
    processor.generate_speeches_df(wc,start_date,end_date,testing)
    print('parsed speeches')
    processor.process_speeches(ngram_min_df,ngram_thresh)
    print('pre-processed speeches')
    processor.make_dtm(dtm_min_df,dtm_max_df)
    processor.save_speeches_df(out_path,filename)
    processor.save_speeches_dtm(out_path,filename)

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options] path chamber omit_tokens out_path")
    parser.add_option('--wc',action='store',type='int',dest='wc',default=50)
    parser.add_option('--sd',action='store',type='string',dest='start_date',default=None)
    parser.add_option('--ed',action='store',type='string',dest='end_date',default=None)
    parser.add_option('--nmin',action='store',type='int',dest='ngram_min_df',default=50)
    parser.add_option('--nt',action='store',type='int',dest='ngram_thresh',default=10)
    parser.add_option('--mindf',action='store',type='int',dest='dtm_min_df',default=30)
    parser.add_option('--maxdf',action='store',type='float',dest='dtm_max_df',default=.3)
    parser.add_option('--f',action='store',type='string',dest='filename',default=None)
    parser.add_option('--t',action='store_true', dest='testing')
    (options,args) = parser.parse_args()
    path, chamber, omit_tokens, out_path = args

    main(path,chamber,omit_tokens,out_path,options.wc, options.start_date,
        options.end_date,options.ngram_min_df,options.ngram_thresh,options.dtm_min_df,
        options.dtm_max_df,options.filename,options.testing)
