import codecs, os, os.path, re
import logging as log
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------

token_pattern = re.compile(r"\b\w\w+\b", re.U)

def custom_tokenizer( s, min_term_length = 2 ):
	"""
	Tokenizer to split text based on any whitespace, keeping only terms of at least a certain length which start with an alphabetic character.
	"""
	return [x.lower() for x in token_pattern.findall(s) if (len(x) >= min_term_length and x[0].isalpha() ) ]

def preprocess( docs, stopwords, min_df = 3, min_term_length = 2, ngram_range = (1,1), apply_tfidf = True, apply_norm = True, tokenizer=custom_tokenizer ):
	"""
	Preprocess a list containing text documents stored as strings.
	"""
	# Build the Vector Space Model, apply TF-IDF and normalize lines to unit length all in one call
	if apply_norm:
		norm_function = "l2"
	else:
		norm_function = None
	tfidf = TfidfVectorizer(stop_words=stopwords, lowercase=True, strip_accents="unicode", tokenizer=tokenizer, use_idf=apply_tfidf, norm=norm_function, min_df = min_df, ngram_range = ngram_range) 
	X = tfidf.fit_transform(docs)
	terms = []
	# store the vocabulary map
	v = tfidf.vocabulary_
	for i in range(len(v)):
		terms.append("")
	for term in v.keys():
		terms[ v[term] ] = term
	return (X,terms)

def load_stopwords( inpath = "text/stopwords.txt" ):
	"""
	Load stopwords from a file into a set.
	"""
	stopwords = set()
	with open(inpath) as f:
		lines = f.readlines()
		for l in lines:
			l = l.strip()
			if len(l) > 0:
				stopwords.add(l)
	return stopwords

# --------------------------------------------------------------

def save_corpus( out_prefix, X, terms, doc_ids ):
	"""
	Save a pre-processed scikit-learn corpus and associated metadata using Joblib.
	"""
	matrix_outpath = "%s.pkl" % out_prefix 
	log.info( "Saving document-term matrix to %s" % matrix_outpath )
	joblib.dump((X,terms,doc_ids), matrix_outpath ) 

def load_corpus( in_path ):
	"""
	Load a pre-processed scikit-learn corpus and associated metadata using Joblib.
	"""
	(X,terms,doc_ids) = joblib.load( in_path )
	return (X, terms, doc_ids)

# --------------------------------------------------------------

def find_documents( root_path ):
	"""
	Find all files in the specified directory and its subdirectories, and store them as strings in a list.
	"""
	filepaths = []
	for dir_path, subFolders, files in os.walk(root_path):
		for filename in files:
			if filename.startswith(".") or filename.startswith("_"):
				continue
			filepath = os.path.join(dir_path,filename)
			filepaths.append( filepath )
	filepaths.sort()
	return filepaths	

# --------------------------------------------------------------

class DocumentBodyGenerator:

	def __init__( self, dir_paths, min_doc_length ):
		self.dir_paths = dir_paths
		self.min_doc_length = min_doc_length

	def __iter__( self ):
		for in_path in self.dir_paths:
			# Find all text files in the directory
			dir_name = os.path.basename( in_path )
			log.info( "- Processing '%s' from %s ..." % (dir_name,in_path) )
			for filepath in find_documents( in_path ):
				doc_id = os.path.splitext( os.path.basename( filepath ) )[0]
				fin = codecs.open(filepath, 'r', encoding="utf8", errors='ignore')
				body = fin.read()
				fin.close()
				if len(body) < self.min_doc_length:
					continue
				yield (doc_id,body)


class DocumentTokenGenerator:

	def __init__( self, dir_paths, min_doc_length, stopwords ):
		self.dir_paths = dir_paths
		self.min_doc_length = min_doc_length
		self.stopwords = stopwords
		self.min_term_length = 2
		self.placeholder = "<stopword>"

	def __iter__( self ):
		bodygen = DocumentBodyGenerator( self.dir_paths, self.min_doc_length )
		for doc_id, body in bodygen:
			body = body.lower().strip()
			tokens = []
			for tok in custom_tokenizer( body, self.min_term_length ):
				if tok in self.stopwords:
					tokens.append( self.placeholder )
				else:
					tokens.append( tok )
			yield tokens
			
