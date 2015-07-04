#!/usr/bin/env python
"""
Simple tool to display topic modeling results generated by NMF, as stored in one or more PKL files.

Requires prettytable:
https://code.google.com/p/prettytable/
"""
import logging as log
from optparse import OptionParser
import unsupervised.util, unsupervised.rankings
from prettytable import PrettyTable

# --------------------------------------------------------------

def format_term_rankings( term_rankings, labels = None, top = 10 ):
	"""
	Format a list of multiple term rankings using PrettyTable.
	"""
	# add header
	header = ["Rank"]
	if labels is None:
		for i in range( len(term_rankings) ):
			header.append("C%02d" % (i+1) )	
	else:
		for label in labels:
			header.append(label)	
	tab = PrettyTable(header)
	# add body
	for pos in range(top):
		row = [ str(pos+1) ]
		for ranking in term_rankings:
			# have we run out of terms?
			if len(ranking) <= pos:
				row.append( "" ) 
			else:
				row.append( ranking[pos] ) 
		tab.add_row( row )
	return tab

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] results_file1 results_file2 ...")
	parser.add_option("-t", "--top", action="store", type="int", dest="top", help="number of top terms to show", default=10)
	(options, args) = parser.parse_args()
	if( len(args) < 1 ):
		parser.error( "Must specify at least one topic modeling results file produced by NMF" )
	log.basicConfig(level=20, format='%(message)s')

	# Load each cached ranking set
	for in_path in args:
		(term_rankings, partition, W, H, terms, labels) = unsupervised.util.load_nmf_results( in_path )
		m = unsupervised.rankings.term_rankings_size( term_rankings )
		log.info( "- Loaded model with %d topics from %s" % (len(term_rankings), in_path) )
		log.info( "Top %d terms for %d topics:" % (options.top,len(term_rankings)) )
		print format_term_rankings( term_rankings, labels, min(options.top,m) )

# --------------------------------------------------------------

if __name__ == "__main__":
	main()