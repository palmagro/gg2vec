from collections import OrderedDict
import json
import pickle as pickle
import os.path
from sklearn import manifold
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show, ColumnDataSource, vplot
from bokeh.models import(
    GMapPlot, Range1d, ColumnDataSource, LinearAxis,
    PanTool, WheelZoomTool, BoxZoomTool, ResetTool, ResizeTool, BoxSelectTool, HoverTool)
from bokeh.charts import Line
from gensim.models import word2vec
import logging
import random
from py2neo import neo4j
import numpy as np
import scipy as scipy
import sys    # sys.setdefaultencoding is cancelled
reload(sys)    # to re-enable sys.setdefaultencoding()
import numpy.linalg as la
from bokeh.charts import BoxPlot
from scipy import signal
import math
from overs import *
from aux import *
from visualization import *
from ZODB import DB
from ZODB.FileStorage import FileStorage
from ZODB.PersistentMapping import PersistentMapping
import transaction
from persistent import Persistent
from persistent.dict import PersistentDict
from persistent.list import PersistentList
from copy import *
from joblib import Parallel, delayed  
import multiprocessing

sys.setdefaultencoding('utf-8')

class node2vec:
    r_deleted = {}
    sentences = {}
    sentences_array = []
    degree = []
    r_types = []
    n_types = []
    r_types_d = []
    r_desv = {}
    n_types_d = []
    m_vectors = []
    m_points = []
    angle_matrix= []
    plotw = 800
    ploth = 500
    mode = "normal"


    def __init__(self,bd,port,user,pss,label,ns,nd,l,m,traversals,iteraciones):
        self.nodes = []
        self.ndim = nd
        self.bd = bd
        self.port = port
        self.user = user
        self.pss = pss
        self.label = label
        self.ns = ns
        self.w_size = l        
        self.mode = m
        self.iteraciones = iteraciones
    
        # Setting up Neo4j DB
        neo4j.authenticate("http://localhost:"+str(self.port), self.user, self.pss)
        self.graph_db = neo4j.GraphDatabaseService("http://neo4j:"+pss+"@localhost:"+str(self.port)+"/db/data/")
        batches = 100

        if not os.path.exists("models/" + self.bd +".npy") or not os.path.exists("models/" + self.bd +"l-degree.npy"):
            print "Conecting to BD..."
            nn = neo4j.CypherQuery(self.graph_db, "match n return count(n) as cuenta1").execute()
            self.numnodes = nn[0].cuenta1
            self.sentences_array = []
            nb = float(self.numnodes/batches)
            count = -1
            self.degree = []
            for i in range(1,int(nb)+1):
                count += 1
                consulta = "match (n)-[r]-(m) where n."+self.label+" <> '' return n,count(r) as d, n."+self.label+", collect(m."+self.label+") as collect skip "+str(batches*count)+" limit "+str(batches)
                cuenta = neo4j.CypherQuery(self.graph_db, consulta).execute()
                print "\r"+str(float((i / nb)*100))+ "%"
                for cuenta1 in cuenta:
                    name = cuenta1['n.'+label].replace(" ","_")
                    context = []
                #Extracting context(relations)
                    for s in cuenta1['collect']:
                        if type(s) is list:
                            for x in s:
                                context.append(str(x).replace(" ","_"))
                        else:
                            if s:
                                context.append(str(s).replace(" ","_"))
                #Extracting contexto(properties)
                    for t in cuenta1['n']:
                        s = cuenta1['n'][t]
                        if type(s) is list:
                            for x in s:
                                context.append(str(x).replace(" ","_"))
                        else:
                            if s:
                                context.append(str(s).replace(" ","_"))
                    if len(context) >= l-1 and cuenta1.d is not None:
                        sentence = context
                        sentence.insert(0,name)
                        self.sentences_array.append(sentence)
                        self.degree.append(cuenta1.d)

            np.save("models/" + self.bd , self.sentences_array)
            np.save("models/" + self.bd +"l-degree", self.degree)   
        else:
            self.sentences_array = np.load("models/" + self.bd +".npy")
            self.degree = np.load("models/" + self.bd +"l-degree.npy")
        for s in self.sentences_array:
            self.sentences[s[0]]=s[1:]
        print "models/" + self.bd +".npy"
        
    def learn(self,m,ts,d,it):
        num_cores = multiprocessing.cpu_count()
        print "numCores = " + str(num_cores)
        self.path = "models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+"l"+m
        if d:
            #el metodo delete_rels elimina las relaciones por las que despues preguntaremos de self.sentences_array antes de entrenar y devuelve el nuevo dump con las relaciones quitadas y una lista de las relaciones quitadas
            self.learn(m,0,False,it)
            self.get_rels([])
            sents,self.r_deleted = delete_rels(self.sentences_array,self.r_types,ts)
            self.path = self.path + "del"+str(ts)+"-"
        else:
            sents = self.sentences_array
        self.path = self.path +str(it)+".npy"
        print "Learning:" + self.path
        print "CCCC!"
        if not os.path.exists(self.path):
            print "Entra"
            entrada = []
            results = Parallel(n_jobs=num_cores, backend="threading")(delayed(generate_sample)(self.mode,sents,self.degree,self.w_size,i) for i in range(1,self.ns))
            for r in results:
                entrada.append(r) 
            self.w2v = word2vec.Word2Vec(entrada, size=self.ndim, window=self.w_size, min_count=1, workers=num_cores,sg=0) 
            self.w2v.save(self.path)
            print "TERMINO"   
        else:
            self.w2v = word2vec.Word2Vec.load(self.path)  
        self.get_nodes()
        self.get_rels([])
        self.delete_props() 

    def get_rels(self,traversals):
        if not os.path.exists("models/" + self.bd+"-trels.p"):
            f = open( "models/" + self.bd+"-trels.p", "w" )
            consulta = neo4j.CypherQuery(self.graph_db, "match (n)-[r]->(m) return n."+self.label+" as s,m."+self.label+" as t ,r,type(r) as tipo,labels(m) as tipot").execute()
            todas = []
            for c in consulta:
                todas.append([c.s,c.tipo,c.t,c.tipot])
            pickle.dump(todas,f)
        else:
            f = open( "models/" + self.bd+"-trels.p", "r" )
            todas = pickle.load(f)
        links = dict()
        for l in todas:
            link = dict()
            if l[0] and l[1] and l[2]:
                link["tipo"] = l[1]
                link["s"] = l[0].replace(" ","_")
                link["t"] = l[2].replace(" ","_")
                link["tipot"] = l[3][0].replace(" ","_")
                if link["s"] in self.w2v and link["t"] in self.w2v:
                    link["v"] = self.w2v[link["t"]] - self.w2v[link["s"]]
                    if not link["tipo"] in links:
                        links[link["tipo"]] = [] 
                    links[link["tipo"]].append(link)
        self.r_types = links

    def r_analysis(self):
        print "Relation Types Analysis"
        if self.r_types == []:
            self.get_rels()    
        self.m_vectors = {}
        for t in self.r_types:
            vectors = []
            rels = self.r_types[t]
            for r in rels:      
                if (r["s"] in self.w2v) and (r["t"] in self.w2v):
                    vectors.append(self.w2v[r["t"]] - self.w2v[r["s"]])
            vector_medio = np.mean(vectors,axis=0)
            self.m_vectors[t] = np.mean(vectors,axis=0)
            media = 0
            for v in vectors:      
                media = media + angle(v,vector_medio) 
            media = media / len(vectors)
            self.r_desv[t] = media
        print "Mean Vector Angles"
        self.angle_matrix= dict()
        for i,t in enumerate(self.r_types):
            self.angle_matrix[t] = dict()    
            for j,x in enumerate(self.r_types):
                self.angle_matrix[t][x] = angle(self.m_vectors[t],self.m_vectors[x])            
                if x not in self.angle_matrix:
                    self.angle_matrix [x]= dict()
                self.angle_matrix[x][t] = angle(self.m_vectors[t],self.m_vectors[x])

    def get_nodes(self):
        if not os.path.exists("models/" + self.bd+"-tnodes.p"):
            f = open( "models/" + self.bd+"-tnodes.p", "w" )
            consulta = neo4j.CypherQuery(self.graph_db, "match (n) return n."+self.label+" as name,labels(n) as tipos").execute()
            nodes = dict()
            for node in consulta:
                if node.name and node.tipos <> []:
                    name = node.name.replace(" ","_")
                    for tipo in node.tipos:
                        if not tipo in nodes:
                            nodes[tipo] = [] 
                        nodes[tipo].append(name)
            self.n_types = nodes
            pickle.dump(nodes,f)
        else:
            f = open( "models/" + self.bd+"-tnodes.p", "r" )
            self.n_types = pickle.load(f)

    def n_analysis(self):
        print "Node Type Analysis"
        if self.n_types == []:
            self.get_nodes()    
        self.m_points = dict()
        self.n_types_d = dict()
        for nt in self.n_types:
            points = []
            for node in self.n_types[nt]:
                if node in self.w2v:
                    points.append(self.w2v[node])
            if len(points) > 0:
                punto_medio = [0] * len(points[0])  
                
                for p in points:
                    for idx,d in enumerate(p):
                        punto_medio[idx] = punto_medio[idx] + d
                for idx,d in enumerate(punto_medio):
                    punto_medio[idx] = punto_medio[idx] / len(points)
                if nt not in self.m_points:
                    self.m_points[nt] = punto_medio
                #print "-------------------"+nt+"-------------------"
                #print "Number of Nodes: "+ str(len(points))
                dev = 0
                for p in points:
                    dev = dev + scipy.spatial.distance.euclidean(punto_medio,p)**2
                dev = math.sqrt((dev / len(points)))
                
                #print "Standard Deviation:"+str(dev)
                if nt not in self.n_types_d:
                    self.n_types_d[nt] = dev
            #print "Variance:"+str(np.var(points))
            
        #print "Distancia entre los puntos medios"
        #for i,t in enumerate(self.m_points):
            #for j,x in enumerate(self.m_points):
                #if i <> j:
                    #print t+" vs. "+x
                    #print scipy.spatial.distance.euclidean(self.m_points[t] , self.m_points[x])                            


    def analysis(self):
        self.n_analysis()
        self.r_analysis()

    def similares(self,nodo,positives,negatives,top_n,filtrado):
        #Version nueva: utilizo las estructuras nodes_pos y nodes_type en un knn de scikit
        clf = neighbors.KNeighborsClassifier(top_n, "uniform",n_jobs=multiprocessing.cpu_count())
        clf.fit(self.nodes_pos, self.nodes_type)
        my_list = clf.kneighbors(positives[0],top_n,False)

        #Version antigua: usaba word2ec por lo que estaba trabajando con todas las propiedades
        my_list = self.w2v.most_similar(positives,negatives,topn=top_n)
        result = []
        for m in my_list:
            if m[0] != nodo:
                result.append(m)
        return result

    def predice(self,nodo,rel,fast,top_n,filtrado):
        if not fast:
            votos = []
            for r in self.r_types[rel]:            
                other = r["s"]
                if(r["s"] == nodo):
                    other = r["t"]
                p2 = neo4j.CypherQuery(self.graph_db, "match (n)-[:"+rel[0]+"]-(m) where n."+label+' = "'+other+'" return m.'+label).execute()
                print p2
                if len(p2) > 0:
                    for p in p2:
                        prop2 = p["m."+label]
                    prop2 = prop2.replace(" ","_")
                    other = other.replace(" ","_")
                    if other in self.w2v and prop2 in self.w2v:
                        prop1 = self.similares([nodo,other],[prop2])[0][0]
                        votos.append(prop1)
            return max(set(votos), key=votos.count)
        if fast:
            sim = self.similares(nodo,[self.w2v[nodo]+self.m_vectors[str(rel)]],[],top_n,filtrado)   
            f = []
            for s in sim:                             
                f.append(s[0])
            if len(f) > 0:
                return f
            else:
                return ""

    def aciertos_rel(self,rel,label,fast,string):
        print "jeje"
        if not os.path.exists("models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+self.mode+"-lpr-"+rel+string+".p"):
            print "ta"
            parcial = 0
            total = 0
            cuenta_misc = 0
            for d in self.r_deleted[rel]:
                print "analizando relacion"
                print rel
                rs = d["s"]
                cuenta_misc += 1
                print rs
                print rs in self.w2v
                print rs in self.sentences
                if rs in self.w2v and not '"' in rs:
                    total = total + 1
                    nbs = self.predice(rs,label,self.r_types1[rel]["t"],rel,fast)
                    if d["t"] in nbs:
                        print "HOLA"
                        print d["t"]
                        print nbs.index(d["t"])
                        parcial += float(1 / float(nbs.index(d["t"])+1 ))
                    print parcial
                    print total
            if total > 0:
                result = float(parcial)/float(total)
            else:
                result = 0
            f = open( "models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+self.mode+"-lpr-"+rel+string+".p", "w" )
            pickle.dump(result,f)
        else:
            f = open( "models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+self.mode+"-lpr-"+rel+string+".p", "r" )
            result = pickle.load(f)
        return result


    def link_prediction_ratio(self):
        ratiosf = {}
        for r in self.r_types:
            ratiosf[r] = self.aciertos_rel(r,self.label,True)

        xname = []
        yname = []
        alpha = []
        color = []
        ratio = []
        names=[]
        for r in self.r_types:
            names.append(r)
            xname.append(r)
            yname.append("Ratio")
            alpha.append(ratiosf[r]/100)
            ratio.append(ratiosf[r])
            color.append('black')
        source = ColumnDataSource(
              
data=dict(
                xname=xname,
                yname=yname,
                colors=color,
                alphas=alpha,
                ratios=ratio
            )
        )
        p = figure(title="Link Prediction Ratios",
            x_axis_location="above", tools="resize,hover,save",
            x_range=xname, y_range=["Ratio"])
        p.rect('xname', 'yname', 0.9, 0.9, source=source,
             color='colors', alpha='alphas', line_color=None)
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "5pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = np.pi/3
        hover = p.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ('link type and method', '@yname, @xname'),
            ('link prediction ratio', '@ratios'),
        ])
        return p
    def ntype(self,n):
        for t in self.n_types:
            if n in self.n_types[t]:
                return t


    def connectZODB(self):
        print "connnecting"
        if not os.path.exists(self.bd+'.fs'):
            self.storage = FileStorage(self.bd+'.fs')
            self.db = DB(self.storage)
            self.connection = self.db.open()
            self.root = self.connection.root()
            self.root = PersistentDict()
        else:
            self.storage = FileStorage(self.bd+'.fs')
            self.db = DB(self.storage)
            self.connection = self.db.open()
            self.root = self.connection.root()

    def disconnectZODB(self):
        print "grabando!"
        transaction.commit()
        self.connection.close()
        self.db.close()
        self.storage.close()
    #Creating nodes_pos dictionary with only nodes vectors (avoiding properties representation) and nodes_target with the type of each node

    def delete_props(self):
        self.nodes_pos = []
        self.nodes_type = []
        self.nodes_name = []
        for t in self.n_types:
            for n in self.n_types[t]:
                if n in self.w2v:
                    self.nodes_pos.append(self.w2v[n])
                    self.nodes_type.append(t)
                    self.nodes_name.append(n)
        print len(self.nodes_pos)
        print len(self.nodes_type)
        self.nodes_pos = list(self.nodes_pos)
        self.nodes_type = list(self.nodes_type)

#Obtenemos el vector medio del traversal solicitado. 
    def get_vtraversal(self,traversal):
        traversals = self.get_traversals(traversal,1)
        total = 0
        suma = "INICIO"
        for t in traversals:
            if t["t"] in self.w2v and t["s"] in self.w2v:
                total += 1
                vector = self.w2v[t["t"]] - self.w2v[t["s"]]
                if suma == "INICIO":        
                    suma = vector
                else:
                    suma += vector
        return suma / total

#Obtenemos una serie de traversals aleatorios del tipo indicado (tantos como indique 0<ts<1) 
    def get_traversals(self,traversal,ts):
        if not os.path.exists("models/" + self.bd+"-trav-" + traversal + ".p"):
            f = open( "models/" + self.bd+"-trav-" + traversal + ".p", "w" )
            consulta = neo4j.CypherQuery(self.graph_db, "match (n)"+traversal+"(m) return n."+self.label+" as s,m."+self.label+" as t ,labels(m) as tipot").execute()
            todas = []
            for c in consulta:
                todas.append({"s":c.s,"t":c.t,"tipot":c.tipot[0]})
            pickle.dump(todas,f)
        else:
            f = open( "models/" + self.bd+"-trav-" + traversal + ".p", "r" )
            todas = pickle.load(f)
        for t in todas:
            t["s"] = t["s"].replace(" ","_")
            t["t"] = t["t"].replace(" ","_")
            t["tipot"] = t["tipot"].replace(" ","_")
        finales = random.sample(todas, int(len(todas)*ts))
        return finales

    def entity_retrieval(self,node,rel_type,target_t):
        temp_pos = []
        temp_name = []
        linkstopredictV = []
        self.r_analysis()
        for idx,e in enumerate(self.nodes_type):
            if e == target_t:
                temp_pos.append(self.nodes_pos[idx])
                temp_name.append(self.nodes_name[idx])
        if len(temp_pos) < 1000:
            ks = len(temp_pos)
        else:
            ks = 1000                                
        clasificador = neighbors.KNeighborsClassifier(ks, "uniform",n_jobs=multiprocessing.cpu_count())
        clasificador.fit(temp_pos, temp_name)
        linkstopredictV.append(self.w2v[node]+self.m_vectors[str(rel_type)])
        nbs = clasificador.kneighbors(linkstopredictV,ks,False)
        result = []
        for e in nbs[0]:
            result.append(temp_name[e])
        return result

