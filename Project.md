The idea of this project is to develop and evaluate a text classification technique, based of the article XMTC.pdf in the parent folder.
What needs to be done is a pipeline is stages, that can do a cascade ranking of labels based on a text.

For the datasets to be used, we want to achieve it using the Eurlex-4k dataset and the AmazonCat-13k. They are problems of classification with a large number of labels, where doing it multi-staged may be usefull.

The idea is to compare different approaches, based on the accuracy, and the computing power needed to achive these results. We want to achieve a good tradeoff between them both.

So we want to do classification with this types of classificators:
BM-25 (CPU Based)
An lightweight/midweight encoder
A robust cross-encoder

An then we will evaluate the results with only them, and then by doing it in stages, like:

BM-25 -> Selects the top 1000/500/100 best results
Encoder -> Ranks the best 100/50 results
Cross-encoder -> Does the final ranking for evaluation.

We want to evaluate the quality of the classification with the same metrics of the article, but we also want to capture statistics on CPU and GPU usage.

The project can be done entirely in Python, using any libraries, by setting up a python environment.

The hardware we have to run this is a 4060ti with 8gb VRAM, the machine has 32GB RAM.
