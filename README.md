# dynamic-nmf

### Summary

Standard topic modeling approaches assume the order of documents does not matter. Not suitable for time-stamped data. In contrast, *dynamic topic modeling* approaches track how language changes and topics evolve over time in a time-stamped corpus. We have developed a two-level approach for dynamic topic modeling via Non-negative Matrix Factorization (NMF), which links together topics identified in snapshots of text sources appearing over time.

Details of this approach are described in the following paper ([Link](http://arxiv.org/abs/1505.07302)):

	Unveiling the Political Agenda of the European Parliament Plenary: A Topical Analysis
	Derek Greene, James P. Cross (2015)
	
This repository contains a Python reference implementation of the above approach.

### Dependencies
Tested with Python 2.7.x and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.8.0](http://www.numpy.org/)
* Required: [scikit-learn >= 0.14](http://scikit-learn.org/stable/)
* Required for utility tools: [prettytable >= 0.7.2](https://code.google.com/p/prettytable/)

### Basic Usage
##### Step 1: Pre-processing
Before applying dynamic topic modeling, the first step is to pre-process the documents from each time window, to produce a *document-term matrix* for those windows. Here, we parse all .txt files in three sub-directories of 'data/sample'. The output files will be stored in the directory 'data'. Note that the final options below indicate that we want to apply TF-IDF term weighting and document length normalization to the documents before writing each matrix.

	python prep-text.py data/sample/month1 data/sample/month2 data/sample/month3 -o data --tfidf --norm

The result of this process will be a collection of Pickle files (*.pkl and *.npy) written to the directory 'data', where the prefix of each corresponds to the name of each time window (e.g. month1, month2 etc).

##### Step 2: Window Topic Modeling 
Once the data has been pre-processed, the next step is to generate the *window topics*, where a topic model is created by applying NMF to each the pre-process data for each time window. For the example data, we apply it to the three months. If we want to use the same number of topics for each window (e.g. 5 topics), we can run the following, where results are written to the directory 'out':
	
	python find-window-topics.py data/month1.pkl data/month2.pkl data/month3.pkl -k 5 -o out

When the process has completed, we can view the descriptiors(i.e. the top ranked terms) for the resulting window topics as follows:

	python display-topics.py out/month1_windowtopics_k05.pkl out/month2_windowtopics_k05.pkl out/month3_windowtopics_k05.pkl

##### Step 3: Dynamic Topic Modeling 
Once the window topics have been created, we combine the results for the time windows to generate the *dynamic topics* that span across multiple time windows. If we want to specify a fixed number of dynamic topics (e.g. 5), we can run the following, where results are written to the directory 'out':

	python find-dynamic-topics.py out/month1_windowtopics_k05.pkl out/month2_windowtopics_k05.pkl out/month3_windowtopics_k05.pkl -k 5 -o out

### Advanced Usage

Details will appear here.

