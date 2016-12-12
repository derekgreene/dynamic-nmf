#!/usr/bin/env python
"""
Tool to pre-process documents contained one or more directories, and export a document-term matrix for each directory.
"""
import os, os.path, sys, codecs
import logging as log
from optparse import OptionParser
import text.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] directory1 directory2 ...")
	parser.add_option("--df", action="store", type="int", dest="min_df", help="minimum number of documents for a term to appear", default=10)
	parser.add_option("--tfidf", action="store_true", dest="apply_tfidf", help="apply TF-IDF term weight to the document-term matrix")
	parser.add_option("--norm", action="store_true", dest="apply_norm", help="apply unit length normalization to the document-term matrix")
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=10)
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="custom stopword file path", default=None)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="output directory (default is current directory)", default=None)
	parser.add_option("--ngram", action="store", type="int", dest="max_ngram", help="maximum ngram range (default is 1, i.e. unigrams only)", default=1)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )	
	log.basicConfig(level=20, format='%(message)s')

	if options.dir_out is None:
		dir_out = os.getcwd()
	else:
		dir_out = options.dir_out	

	# Load required stopwords
	if options.stoplist_file is None:
		stopwords = text.util.load_stopwords()
	else:
		log.info( "Using custom stopwords from %s" % options.stoplist_file )
		stopwords = text.util.load_stopwords( options.stoplist_file )

	# Process each directory
	for in_path in args:
		dir_name = os.path.basename( in_path )
		# Read content of all documents in the directory
		docgen = text.util.DocumentBodyGenerator( [in_path], options.min_doc_length )
		docs = []
		doc_ids = []
		for doc_id, body in docgen:
			docs.append(body)	
			doc_ids.append(doc_id)	
		log.info( "Found %d documents to parse" % len(docs) )

		# Pre-process the documents
		log.info( "Pre-processing documents (%d stopwords, tfidf=%s, normalize=%s, min_df=%d, max_ngram=%d) ..." % (len(stopwords), options.apply_tfidf, options.apply_norm, options.min_df, options.max_ngram ) )
		(X,terms) = text.util.preprocess( docs, stopwords, min_df = options.min_df, apply_tfidf = options.apply_tfidf, apply_norm = options.apply_norm, ngram_range = (1,options.max_ngram) )
		log.info( "Created %dx%d document-term matrix" % X.shape )

		# Save the pre-processed documents
		out_prefix = os.path.join( dir_out, dir_name )
		text.util.save_corpus( out_prefix, X, terms, doc_ids )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
