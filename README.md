# node2vec

Learning Neo4j DB content embedding via Deep Learning.

from node2vec import *

n2v = node2vec(bd_name,bd_url,bd_username,bd_password,label,num_sentences,num_dims,windows_size,mode)

Generates a embedding representation of the Neo4j Database located in bd_url

The library offers some tools to analyze a Neo4j DB using Deep Learning. 

n2v.all_figure()

n2v.nodes_figure()

n2v.links_figure()

n2v.angles_figure()

In addition, link prediction and entity resolution tools are provided too by node2vec library.
