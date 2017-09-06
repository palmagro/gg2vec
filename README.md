## Learning Neo4j DB content embedding using word2vec architectures

"In this work a new machine learning approach to the study of Generalized Graphs as semantic data structures is presented. It shows how vector representations that maintain semantic and topological features of the original data can be obtained from neural encoding architectures and considering the topological properties of the graph. Also, semantic features of these new representations are tested by using some machine learning tasks and new directions on efficient link discovery methodologies on large relational datasets are investigated."

We present **gg2vec**, a python library to perform vector space embeddings of generalized graphs stored in a Neo4j DB. Next we we will demonstrate the power of the library, in the example we are using the Movie Database (available in https://neo4j.com/developer/movie-database/ to allow reproducibility). 

```python
from gg2vec import *
from experiment import *
from composite_experiment import *

cine = gg2vec("cine",7474,"neo4j","******","name",400000,200,6,"normal",[],1)
#gg2vec(bd_name,bd_port,bd_username,bd_password,label,num_sentences,num_dim,windows_size,mode,traversals,num_iterations)
cine.learn("normal",0.5,False,0)
#learn(mode,freq_of_del_links,del_links?,n_of_repetitions)
```
Generates a num_dim dimension embedding representation of the indicated Neo4j Database. This method trains a CBOW neural network with pairs word-context (w,C) where w is a node and C is a window of his context (properties and neighbours). The parameter "mode" can be "normal" or "degree". Normal mode generates random (w,C) pairs. Degree mode generates (w,C) where the probability to generate a pair (w,C) is proportionally to degree of node w.

The library visualization.py offers some tools to visualize a Neo4j DB embedding. 

```python
from visualization import *
all_figure(N2v,[0.02,0.02,1],["Actor","Movie","Genre"],False)
#all_figure()
```
![png](https://s26.postimg.org/71ge1l8ah/cine_all_nodes_no_names.png)


```python
from visualization import *
all_links_figure(cine,[0.03],["GENRE"],True,10)
#all_links_figure()
```
![png](https://s26.postimg.org/tc491k5kp/cine_all_nodes.png)
```python
show((all_figure(N2v,[0.02,0.02,1],["Actor","Movie","Genre"],False)))
#all_figure()
```
![png](https://s26.postimg.org/9y3f1ve49/cine_all_links.png)

**gg2vec** libary allows to get the closest node to a given one fulfilling some restrictions given by the user. In addition, entity retrieval and long distance query tools are provided too by **gg2vec** library.

n2v.all_figure()

n2v.nodes_figure()

n2v.links_figure()

n2v.angles_figure()

In addition, link prediction and entity resolution tools are provided too by node2vec library.

Lets watch an example (Asterix)...
