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
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=50)
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="custom stopword file path", default=None)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="output directory (default is current directory)", default=None)
	parser.add_option('-d','--debug',type="int",help="Level of log output; 0 is less, 5 is all", default=3)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one directory" )	
	log.basicConfig(level=max(50 - (options.debug * 10), 10), format='%(message)s')

	if options.dir_out is None:
		dir_out = os.getcwd()
	else:
		dir_out = options.dir_out	

	# Load required stopwords
	if options.stoplist_file is None:
		stopwords = text.util.load_stopwords()
	else:
		log.info( "Using custom stopwords from", options.stoplist_file )
		stopwords = text.util.load_stopwords( options.stoplist_file )

	# Process each directory
	for in_path in args:
		# Find all text files in the directory
		dir_name = os.path.basename( in_path )
		log.info( "- Processing '%s' from %s ..." % (dir_name,in_path) )
		filepaths = []
		for fpath in text.util.find_documents( in_path ):
			filepaths.append( fpath )
		log.info( "Found %d documents to parse" % len(filepaths) )

		# Read content of all documents in the directory
		docs = []
		doc_ids = []
		for filepath in filepaths:
			# TODO: remove
			if len(doc_ids) > 5000:
				break
			doc_id = os.path.splitext( os.path.basename( filepath ) )[0]
			# read body text
			log.debug( "Reading text from %s ..." % filepath )
			fin = codecs.open(filepath, 'r', encoding="utf8", errors='ignore')
			body = fin.read()
			fin.close()
			if len(body) < options.min_doc_length:
				continue
			docs.append(body)	
			doc_ids.append(doc_id)	
		log.info( "Kept %d documents. Skipped %d documents with length < %d" % ( len(docs), len(filepaths)-len(docs), options.min_doc_length ) )

		# Pre-process the documents
		log.info( "Pre-processing documents (%d stopwords, tfidf=%s, normalize=%s, min_df=%d) ..." % (len(stopwords), options.apply_tfidf, options.apply_norm, options.min_df) )
		(X,terms) = text.util.preprocess( docs, stopwords, min_df = options.min_df, apply_tfidf = options.apply_tfidf, apply_norm = options.apply_norm )
		log.info( "Created %dx%d document-term matrix" % X.shape )

		# Save the pre-processed documents
		out_prefix = os.path.join( dir_out, dir_name )
		text.util.save_corpus( out_prefix, X, terms, doc_ids )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
