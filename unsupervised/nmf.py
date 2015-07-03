import numpy as np

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# --------------------------------------------------------------

class SklNMF:
	"""
	Wrapper class backed by the scikit-learn package NMF implementation.
	"""
	def __init__( self, max_iters = 100, init_strategy = "random" ):
		self.max_iters = 100
		self.init_strategy = init_strategy
		self.W = None
		self.H = None

	def apply( self, X, k = 2 ):
		"""
		Apply NMF to the specified document-term matrix X.
		"""
		from sklearn import decomposition
		self.W = None
		self.H = None
		model = decomposition.NMF(init=self.init_strategy, n_components=k, max_iter=self.max_iters)
		self.W = model.fit_transform(X)
		self.H = model.components_			
		
	def rank_terms( self, topic_index, top = -1 ):
		"""
		Return the top ranked terms for the specified topic, generated during the last NMF run.
		"""
		if self.H is None:
			raise ValueError("No results for previous run available")
		# NB: reverse
		top_indices = np.argsort( self.H[topic_index,:] )[::-1]
		# truncate if necessary
		if top < 1 or top > len(top_indices):
			return top_indices
		return top_indices[0:top]

	def generate_partition( self ):
		if self.W is None:
			raise ValueError("No results for previous run available")
		return np.argmax( self.W, axis = 1 ).flatten().tolist()		

class NimfaNMF:
	"""
	Wrapper class backed by the Nimfa package NMF implementation.
	"""
	def __init__( self, max_iters = 100, init_strategy = "random", update = "euclidean", method = "lsnmf" ):
		self.max_iters = max_iters
		self.init_strategy = init_strategy
		self.W = None
		self.H = None
		self.update = update
		self.test_conv = 10
		self.method = method

	def apply( self, X, k = 2 ):
		"""
		Apply NMF to the specified document-term matrix X.
		"""
		import nimfa
		self.W = None
		self.H = None
		initialize_only = self.max_iters < 1
		if self.update == "euclidean":
			objective = "fro"
		else:
			objective = "div"
		alg = nimfa.mf(X, method = self.method, max_iter = self.max_iters, rank = k, seed = self.init_strategy, update = self.update, objective = objective, test_conv = self.test_conv ) 
		res = nimfa.mf_run(alg)
		# TODO: fix
		try:
			self.W = res.basis().todense() 
			self.H = res.coef().todense()
		except:
			self.W = res.basis()
			self.H = res.coef()
		# last number of iterations
		self.n_iter = res.n_iter

	def rank_terms( self, topic_index, top = -1 ):
		"""
		Return the top ranked terms for the specified topic, generated during the last NMF run.
		"""
		if self.H is None:
			raise ValueError("No results for previous run available")
		h = np.array( self.H[topic_index,:] ).flatten()
		# NB: reverse ordering
		top_indices = np.argsort(h)[::-1]
		# truncate
		if top < 1 or top > len(top_indices):
			return top_indices
		return top_indices[0:top]

	def generate_partition( self ):
		if self.W is None:
			raise ValueError("No results for previous run available")
		return np.argmax( self.W, axis = 1 ).flatten().tolist()[0]


