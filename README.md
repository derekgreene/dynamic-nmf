# dynamic-nmf

### Summary

Standard topic modeling approaches assume the order of documents does not matter. Not suitable for time-stamped data. In contrast, *dynamic topic modeling* approaches track how language changes and topics evolve over time in a time-stamped corpus. We have developed a two-level approach for dynamic topic modeling via Non-negative Matrix Factorization (NMF), which links together topics identified in snapshots of text sources appearing over time.

Details of this approach are described in the following paper:

	Unveiling the Political Agenda of the European Parliament Plenary: A Topical Analysis (2015)
	Derek Greene, James P. Cross
	http://arxiv.org/abs/1505.07302
	
This repository contains a Python reference implementation of the above approach.

### Dependencies
Tested with Python 2.7.x and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.8.0](http://www.numpy.org/)
* Required: [scikit-learn >= 0.14](http://scikit-learn.org/stable/)
* Required for utility tools: [prettytable >= 0.7.2](https://code.google.com/p/prettytable/)

### Basic Usage
Details will appear here.
