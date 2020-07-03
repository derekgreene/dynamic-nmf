#!/usr/bin/env python
"""
Tool to pre-process documents contained one or more directories, and build a Word2Vec model. 

This implementation requires Gensim. For documentation regarding the various parameters, see:
https://radimrehurek.com/gensim/models/word2vec.html
"""
import os, os.path, sys
import logging as log
from optparse import OptionParser
import gensim
import text.util

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] directory1 directory2 ...")
	parser.add_option("-s", action="store", type="string", dest="stoplist_file", help="custom stopword file path", default=None)
	parser.add_option("--df", action="store", type="int", dest="min_df", help="minimum number of documents for a term to appear", default=10)
	parser.add_option("--minlen", action="store", type="int", dest="min_doc_length", help="minimum document length (in characters)", default=50)
	parser.add_option("-d","--dimensions", action="store", type="int", dest="dimensions", help="the dimensionality of the word vectors", default=500)
	parser.add_option("--window", action="store", type="int", dest="w2v_window", 
		help="the maximum distance for Word2Vec to use between the current and predicted word within a sentence", default=5)
	parser.add_option("--lem", action="store", type="string", dest="lem_file", help="lemmatizer dictionary file path", default=None)
	parser.add_option("-o","--outdir", action="store", type="string", dest="dir_out", help="output directory (default is current directory)", default=None)
	parser.add_option("-m", action="store", type="string", dest="model_type", help="type of word embedding model to build (sg or cbow)", default="sg")
	parser.add_option("--debug", action="store_true", dest="debug", help="enable debugging information", default=False)
	# Parse command line arguments
	(options, args) = parser.parse_args()
	if len(args) < 1:
		parser.error("Must specify at least one directory of file path")	
	# control level of log output
	log_level = log.DEBUG if options.debug else log.INFO
	log.basicConfig(level=log_level, format='%(message)s')

	if options.dir_out is None:
		dir_out = os.getcwd()
	else:
		dir_out = options.dir_out	

	# Load required stopwords
	if options.stoplist_file is None:
		stopwords = text.util.load_stopwords()
	else:
		log.info("Using custom stopwords from %s" % options.stoplist_file)
		stopwords = text.util.load_stopwords( options.stoplist_file )
	if stopwords is None:
		stopwords = set()
		log.info("No stopword list available")
	else:
		log.info("Loaded %d stopwords" % len(stopwords))

	# Load lemmatization dictionary, if specified
	lemmatizer = None
	if not options.lem_file is None:
		log.info("Loading lemmatization dictionary from %s ..." % options.lem_file)
		lemmatizer = text.util.DictLemmatizer(options.lem_file)
		# add any missing lemmatized stopwords
		extra_stopwords = set()
		for stopword in stopwords:
			extra_stopwords.add(lemmatizer.apply(stopword))
		stopwords = extra_stopwords
		log.info("Using %d stopwords after lemmatization" % len(stopwords))

	# Process all specified directories
	docgen = text.util.DocumentTokenGenerator(args, options.min_doc_length, stopwords, lemmatizer)

	# Build the Word2Vec model from the documents that we have found
	log.info( "Building Word2vec %s model..." % options.model_type )
	if options.model_type == "cbow":
		model = gensim.models.Word2Vec(docgen, size=options.dimensions, min_count=options.min_df, window=options.w2v_window, workers=4, sg = 0)
	elif options.model_type == "sg":
		model = gensim.models.Word2Vec(docgen, size=options.dimensions, min_count=options.min_df, window=options.w2v_window, workers=4, sg = 1)
	else:
		log.error("Unknown model type '%s'" % options.model_type )
		sys.exit(1)
	log.info( "Built model: %s" % model )

	# Save the Word2Vec model
	model_out_path = os.path.join( dir_out, "w2v-model.bin" )
	log.info( "Writing model to %s ..." % model_out_path )
	model.save(model_out_path)
			
# --------------------------------------------------------------

if __name__ == "__main__":
	main()
