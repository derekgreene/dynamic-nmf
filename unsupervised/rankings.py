"""
Utility function for working with term rankings.
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

