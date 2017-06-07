#!/usr/bin/env python
"""
Merge document partitions from window topics to create an overall document partition for a dynamic model.

Usage: 
python create-dynamic-partition.py -o out/dynamic-combined.pkl out/dynamictopics_k05.pkl out/month1_windowtopics_k05.pkl out/month2_windowtopics_k08.pkl out/month3_windowtopics_k08.pkl
"""
import os, sys
import logging as log
from optparse import OptionParser
from prettytable import PrettyTable
import unsupervised.nmf, unsupervised.rankings

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] dynamic_topics window_topics1 window_topics2...")
	parser.add_option("-o", action="store", type="string", dest="out_path", help="output path", default=None)
	(options, args) = parser.parse_args()
	if( len(args) < 3 ):
		parser.error( "Must specify at least a dynamic topic file, followed by two or more window topic files (in order of time window)" )
	log.basicConfig(level=20, format='%(message)s')

	# Load dynamic results: (doc_ids, terms, term_rankings, partition, W, H, labels)
	dynamic_in_path = args[0]
	dynamic_res = unsupervised.nmf.load_nmf_results( dynamic_in_path )
	dynamic_k = len(dynamic_res[2])
	dynamic_partition = dynamic_res[3]
	log.info( "Loaded model with %d dynamic topics from %s" % (dynamic_k, dynamic_in_path) )
	
	# Create a map of window topic label -> dynamic topic
	assigned_window_map = {}
	dynamic_partition = dynamic_res[3]
	for idx, window_topic_label in enumerate(dynamic_res[0]):
		assigned_window_map[window_topic_label] = dynamic_partition[idx]

	all_partition = []
	all_doc_ids = []

	# Process each window topic model
	window_num = 0
	for in_path in args[1:]:
		window_num += 1
		log.info( "Reading window topics for window %d from %s ..." % ( window_num, in_path ) )
		# Load window results: (doc_ids, terms, term_rankings, partition, W, H, labels)
		window_res = unsupervised.nmf.load_nmf_results( in_path )
		window_doc_ids = window_res[0]
		window_k = len(window_res[2])
		window_partition = window_res[3]
		for window_topic_idx, window_topic_label in enumerate(window_res[6]):
			dynamic_topic_idx = assigned_window_map[window_topic_label]
			for i, doc_id in enumerate(window_doc_ids):
				if window_partition[i] == window_topic_idx:
					all_doc_ids.append( doc_id )
					all_partition.append( dynamic_topic_idx )

	log.info("Created overall partition covering %d documents" % len(all_doc_ids) )

	# TODO: fix W and H
	if options.out_path is None:
		results_out_path = "dynamic-combined.pkl"
	else:
		results_out_path = options.out_path
	unsupervised.nmf.save_nmf_results( results_out_path, all_doc_ids, dynamic_res[1], dynamic_res[2], all_partition, None, None, dynamic_res[6] )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
 
