"""
Utility functions for working with term rankings.
"""

def term_rankings_size( term_rankings ):
	"""
	Return the number of terms covered by a list of multiple term rankings.
	"""
	m = 0
	for ranking in term_rankings:
		if m == 0:
			m = len(ranking)
		else:
			m = min( len(ranking), m ) 
	return m

def truncate_term_rankings( orig_rankings, top ):
	"""
	Truncate a list of multiple term rankings to the specified length.
	"""
	if top < 1:
		return orig_rankings
	trunc_rankings = []
	for ranking in orig_rankings:
		trunc_rankings.append( ranking[0:min(len(ranking),top)] )
	return trunc_rankings


def format_term_rankings( term_rankings, labels = None, top = 10 ):
	"""
	Format a list of multiple term rankings using PrettyTable.
	"""
	from prettytable import PrettyTable
	# add header
	header = ["Rank"]
	if labels is None:
		for i in range( len(term_rankings) ):
			header.append("C%02d" % (i+1) )	
	else:
		for label in labels:
			header.append(label)	
	tab = PrettyTable(header)
	tab.align["Rank"] = "r"
	for label in header[1:]:
		tab.align[label] = "l"
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

def format_term_rankings_long( term_rankings, labels = None, top = 10 ):
	"""
	Format a list of multiple term rankings using lists.
	"""
	if labels is None:
		labels = []
		for i in range( len(term_rankings) ):
			labels.append("C%02d" % (i+1) )	
	max_label_len = 0
	for label in labels:
		max_label_len = max(max_label_len,len(label))
	max_label_len += 1

	s = ""
	for i, label in enumerate(labels):
		s += label.ljust(max_label_len)
		s += ": "
		sterms = ""
		for term in term_rankings[i][0:top]:
			if len(sterms) > 0:
				sterms += ", "
			sterms += term
		s += sterms + "\n"
	return s
