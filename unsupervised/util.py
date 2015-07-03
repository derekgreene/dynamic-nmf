import numpy as np
import logging as log
from scipy import sparse as sp
# note that we use the scikit-learn bundled version of joblib
from sklearn.externals import joblib

# --------------------------------------------------------------

def save_nmf_results( out_path, term_rankings, partition, W, H, terms, topic_labels=None ):
	"""
	Save output of NMF using Joblib.
	"""
	# no labels? generate some standard ones
	if topic_labels is None:
		topic_labels = []
		for i in range( len(term_rankings) ):
			topic_labels.append( "C%02d" % (i+1) )
	log.info( "Saving NMF results to %s" % out_path )
	joblib.dump((term_rankings, partition, W, H, terms, topic_labels), out_path ) 

def load_nmf_results( in_path ):
	"""
	Load NMF results using Joblib.
	"""
	(term_rankings, partition, W, H, terms, labels) = joblib.load( in_path )
	return (term_rankings, partition, W, H, terms, labels)

