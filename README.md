## Learning Neo4j DB content embedding using neural encoders

"In this work a new machine learning approach to the study of Generalized Graphs as semantic data structures is presented. It shows how vector representations that maintain semantic and topological features of the original data can be obtained from neural encoding architectures and considering the topological properties of the graph. Also, semantic features of these new representations are tested by using some machine learning tasks and new directions on efficient link discovery methodologies on large relational datasets are investigated."

We present **gg2vec**, a python library to perform vector space embeddings of generalized graphs stored in a Neo4j DB. **gg2vec** uses word2vec neuroal encoder architectures (https://arxiv.org/abs/1301.3781) to obtain vector representations of the elements of a Neo4j graph. Next we we will demonstrate the capabilities of the library using the Movie Database (available in https://neo4j.com/developer/movie-database/ to allow reproducibility) as example data. 

```python
from gg2vec import *
from experiment import *
from composite_experiment import *

cine = gg2vec("cine",7474,"neo4j","******","name",400000,200,6,"normal",[],1)
#gg2vec(bd_name,bd_port,bd_username,bd_password,label,num_sentences,num_dim,windows_size,mode,traversals,num_iterations)
cine.learn("normal",0.5,False,0)
#gg2vec.learn(mode,freq_of_del_links,del_links?,n_of_repetitions)
```
Generates a num_dim dimension embedding representation of the indicated Neo4j Database. This method trains a CBOW neural network with pairs word-context (w,C) where w is a node and C is a window of his context (properties and neighbours). The "mode" parameter is set as "normal" or "degree". "normal" mode generates random (w,C) pairs. "degree" mode generates (w,C) where the probability of generating a pair (w,C) is proportional to the degree of node w.

The library visualization.py offers some tools to visualize a Neo4j DB embedding. 

```python
from visualization import *
all_figure(cine,[0.02,0.02,1],["Actor","Movie","Genre"],False)
#all_figure(bd_name,visible_rate_array,array_of_node_types,legend?)
```
![png](https://s26.postimg.org/71ge1l8ah/cine_all_nodes_no_names.png)


```python
from visualization import *
all_figure(cine,[1],["Genre"],True)
#all_figure(bd_name,visible_rate_array,array_of_node_types,legend?)
```
![png](https://s26.postimg.org/tc491k5kp/cine_all_nodes.png)
```python
all_links_figure(cine,[0.03],["GENRE"],True,10)
#all_links_figure(bd_name,visible_rate_array,array_of_link_types,legend?,treshold)
```
![png](https://s26.postimg.org/9y3f1ve49/cine_all_links.png)

A modification on CBOW architecture implementation of Gensim toolkit (https://radimrehurek.com/gensim) (version 0.12.4) is necessary in order to allow the system to work properly. The library performs such modification automatically. For further information please contact me.
