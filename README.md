## Learning Neo4j DB content embedding using word2vec architectures

"In this work a new machine learning approach to the study of Generalized Graphs as semantic data structures is presented. It shows how vector representations that maintain semantic and topological features of the original data can be obtained from neural encoding architectures and considering the topological properties of the graph. Also, semantic features of these new representations are tested by using some machine learning tasks and new directions on efficient link discovery methodologies on large relational datasets are investigated."

We present **gg2vec**, a python library to perform vector space embeddings of generalized graphs stored in a Neo4j DB. Next we we will demonstrate the power of the library, in the next eample we are using the Movie Database (available in https://neo4j.com/developer/movie-database/ to allow reproducibility). 



```python
from node2vec import *

n2v = node2vec(bd_name,bd_url,bd_username,bd_password,label,num_sentences,num_dim,windows_size,mode)

Generates a num_dim dimension embedding representation of the Neo4j Database located in bd_url. This method trains a neural network with pairs word-context (w,C) where w is a node and C is a window of his context (properties and neighbours). The last parameter "mode" can be "normal" or "degree". Normal mode generates random (w,C) pairs. Degree mode generates (w,C) where the probability to generate a pair (w,C) is proportionally to degree of node w.

The library offers some tools to analyze a Neo4j DB using Deep Learning. 

n2v.all_figure()

n2v.nodes_figure()

n2v.links_figure()

n2v.angles_figure()

In addition, link prediction and entity resolution tools are provided too by node2vec library.

Lets watch an example (Asterix)...
