# node2vec

Learning Neo4j DB content embedding via Deep Learning.

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
