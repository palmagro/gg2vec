from collections import OrderedDict
import json
import pickle as pickle
from overs import *
from aux import *
import os.path
from sklearn import manifold
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show, ColumnDataSource, vplot
from bokeh.models import(
    GMapPlot, Range1d, ColumnDataSource, LinearAxis,
    PanTool, WheelZoomTool, BoxZoomTool, ResetTool, ResizeTool, BoxSelectTool, HoverTool,
    BoxSelectionOverlay, GMapOptions,
    NumeralTickFormatter, PrintfTickFormatter)
from bokeh.charts import Line
from gensim.models import word2vec
import logging
import random
from py2neo import neo4j
import numpy as np
import scipy as scipy
import sys    # sys.setdefaultencoding is cancelled by site.py
reload(sys)    # to re-enable sys.setdefaultencoding()
import numpy.linalg as la
from bokeh.charts import BoxPlot
from scipy import signal
import math


sys.setdefaultencoding('utf-8')

class node2vec:
    sentences = {}
    degree = []
    r_types = []
    n_types = []
    r_types_d = []
    n_types_d = []
    m_vectors = []
    m_points = []
    angle_matrix= []
    plotw = 800
    ploth = 500
    mode = "normal"


    def __init__(self,bd,port,user,pss,label,ns,nd,l,m):
        self.ndim = nd
        self.bd = bd
        self.port = port
        self.user = user
        self.pss = pss
        self.label = label
        self.ns = ns
        self.w_size = l        
        self.mode = m
        neo4j.authenticate("http://localhost:"+str(self.port), self.user, self.pss)
        self.graph_db = neo4j.GraphDatabaseService("http://neo4j:"+pss+"@localhost:"+str(self.port)+"/db/data/")
        batches = 100
        if not os.path.exists("models/" + self.bd +".npy") or not os.path.exists("models/" + self.bd +"l-degree.npy"):
            print "Conecting to BD..."
            nn = neo4j.CypherQuery(self.graph_db, "match n return count(n) as cuenta1").execute()
            self.numnodes = nn[0].cuenta1
            sentences = []
            nb = float(self.numnodes/batches)
            count = -1
            self.degree = []
            for i in range(1,int(nb)+1):
                count += 1
                consulta = "match (n)-[r]-(m) where n."+self.label+" <> '' return n,count(r) as d, n."+self.label+", collect(m."+self.label+") as collect skip "+str(self.batches*count)+" limit "+str(self.batches)
                cuenta = neo4j.CypherQuery(self.graph_db, consulta).execute()
                print "\r"+str(float((i / nb)*100))+ "%"
                for cuenta1 in cuenta:
                    name = cuenta1['n.'+label].replace(" ","_")
                    context = []
                #Extraemos contexto/relaciones
                    for s in cuenta1['collect']:
                        if type(s) is list:
                            for x in s:
                                context.append(str(x).replace(" ","_"))
                        else:
                            if s:
                                context.append(str(s).replace(" ","_"))
                #Extraemos contexto/propiedades    
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
                        sentences.append(sentence)
                        self.degree.append(cuenta1.d)

            np.save("models/" + self.bd , sentences)
            np.save("models/" + self.bd +"l-degree", self.degree)   
        else:
            sentences = np.load("models/" + self.bd +".npy")
            self.degree = np.load("models/" + self.bd +"l-degree.npy")
        for s in sentences:
            self.sentences[s[0]]=s[1:]
        print "asd"
        print "models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+"l"+m+".npy"
        if not os.path.exists("models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+"l"+m+".npy"):
            entrada = []
            for i in range(1,self.ns):
                if m == "degree":
                    s = sentences[weighted_choice(self.degree)]
                else:
                    s = np.random.choice(sentences)
                s = eval(str(s))               
                a = s[0] 
                b = sample_wr(s[1:],l)
                b.insert(0,a)
                entrada.append(b)
            self.w2v = word2vec.Word2Vec(entrada, size=nd, window=l, min_count=1, workers=4,sg=0)        
            self.w2v.save("models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+"l"+m+".npy")
                     
        else:
            self.w2v = word2vec.Word2Vec.load("models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+"l"+m+".npy")
        print "Terminado:" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+"l"
        self.analysis()

    def get_rels(self):
        if not os.path.exists("models/" + self.bd+"-trels.p"):
            f = open( "models/" + self.bd+"-trels.p", "w" )
            consulta = neo4j.CypherQuery(self.graph_db, "match (n)-[r]->(m) return n."+self.label+" as s,m."+self.label+" as t ,r,type(r) as tipo").execute()
            rels = dict()
            for r in consulta:
                rel = dict()
                if r.s and r.t:
                    rel["s"] = unicode(r.s.replace(" ","_"))
                    rel["t"] = unicode(r.t.replace(" ","_"))
                    if not r.tipo in rels:
                        rels[r.tipo] = [] 
                    rels[r.tipo].append(rel)
            self.r_types = rels
            print "asasfafs"
            f.write(json.dumps(rels))
        else:
            f = open( "models/" + self.bd+"-trels.p", "r" )
            self.r_types = json.loads(f.read())
        if not os.path.exists("models/" + self.bd+"-trels1.p"):
            f = open( "models/" + self.bd+"-trels1.p", "w" )
            consulta = neo4j.CypherQuery(self.graph_db, "MATCH (a)-[r]->(b) WHERE labels(a) <> [] AND labels(b) <> [] RETURN DISTINCT head(labels(a)) AS This, type(r) as To, head(labels(b)) AS That ").execute()
            rels = dict()
            for r in consulta:
                rel = dict()
                rel["s"] = unicode(r.This.replace(" ","_"))
                rel["r"] = unicode(r.To.replace(" ","_"))
                rel["t"] = unicode(r.That.replace(" ","_"))
                if rel["r"] not in rels:
                    rels[rel["r"]] = rel
            self.r_types1 = rels
            f.write(json.dumps(rels))
        else:
            f = open( "models/" + self.bd+"-trels1.p", "r" )
            self.r_types1 = json.loads(f.read())


    def r_analysis(self):
        print "Analisis de los tipos de relaciones"
        if self.r_types == []:
            self.get_rels()    
        self.m_vectors = {}
        for t in self.r_types:
            vectors = []
            rels = self.r_types[t]
            print "------------"+ t+"------------"
            print "Number of Relations: "+ str(len(rels))
            for r in rels:      
                if (r["s"] in self.w2v) and (r["t"] in self.w2v):
                    vectors.append(self.w2v[r["t"]] - self.w2v[r["s"]])
            print "Number of Good Relations: "+ str(len(vectors))
            vector_medio = np.mean(vectors,axis=0)
            self.m_vectors[t] = np.mean(vectors,axis=0)
#            self.m_vectors.append(np.mean(vectors,axis=0))
            media = 0
            for v in vectors:      
                media = media + angle(v,vector_medio) 
            media = media / len(vectors)
            print "Mean Angle Deviation:" +  str(media)
            self.r_types_d.append(media)
        print "Angulos entre vectores medios"
        self.angle_matrix= dict()
        for i,t in enumerate(self.r_types):
            self.angle_matrix[t] = dict()    
            for j,x in enumerate(self.r_types):
                self.angle_matrix[t][x] = angle(self.m_vectors[t],self.m_vectors[x])            
                if x not in self.angle_matrix:
                    self.angle_matrix [x]= dict()
                self.angle_matrix[x][t] = angle(self.m_vectors[t],self.m_vectors[x])
                if i <> j:
                    print t+" vs. "+x
                    print angle(self.m_vectors[t],self.m_vectors[x])            

    def get_nodes(self):
        if not os.path.exists("models/" + self.bd+"-tnodes.p"):
            f = open( "models/" + self.bd+"-tnodes.p", "w" )
            consulta = neo4j.CypherQuery(self.graph_db, "match (n) return n."+self.label+" as name,labels(n) as tipos").execute()
            nodes = dict()
            for node in consulta:
                if node.name and node.tipos <> []:
                    tipo = node.tipos[0]
                    name = node.name.replace(" ","_")
                    if not tipo in nodes:
                        nodes[tipo] = [] 
                    nodes[tipo].append(name)
            self.n_types = nodes
            pickle.dump(nodes,f)
        else:
            f = open( "models/" + self.bd+"-tnodes.p", "r" )
            self.n_types = pickle.load(f)

    def n_analysis(self):
        print "Analisis de los tipos de nodos"
        if self.n_types == []:
            self.get_nodes()    
        self.m_points = dict()
        self.n_types_d = dict()
        for nt in self.n_types:
            points = []
            for node in self.n_types[nt]:
                if node in self.w2v:
                    points.append(self.w2v[node])
            punto_medio = [0] * len(points[0])  
            for p in points:
                for idx,d in enumerate(p):
                    punto_medio[idx] = punto_medio[idx] + d
            for idx,d in enumerate(punto_medio):
                punto_medio[idx] = punto_medio[idx] / len(points)
            if nt not in self.m_points:
                self.m_points[nt] = punto_medio
            print "-------------------"+nt+"-------------------"
            print "Number of Nodes: "+ str(len(points))
            dev = 0
            for p in points:
                dev = dev + scipy.spatial.distance.euclidean(punto_medio,p)**2
            dev = math.sqrt((dev / len(points)))
            
            print "Standard Deviation:"+str(dev)
            if nt not in self.n_types_d:
                self.n_types_d[nt] = dev
            #if nt not in self.n_types_d:
                #self.n_types_d[nt] = scipy.std(points,axis=0)
            #if nt not in self.n_types_d:
            #    self.n_types_d[nt] = np.mean(scipy.spatial.distance.pdist(points))
            print "Variance:"+str(np.var(points))
            
        print "Distancia entre los puntos medios"
        for i,t in enumerate(self.m_points):
            for j,x in enumerate(self.m_points):
                if i <> j:
                    print t+" vs. "+x
                    print scipy.spatial.distance.euclidean(self.m_points[t] , self.m_points[x])                            

    def analysis(self):
        self.n_analysis()
        self.r_analysis()

    def all_figure(self):
        pal = pallete("nodes") 
        #mds = manifold.TSNE(n_components=2)
        mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
        X = []
        Y = []
        C = []
        for idx,nt in enumerate(self.n_types):
            for n in self.n_types[nt]:
                if n in self.w2v:
                    X.append(self.w2v[n])
                    Y.append(n)
                    C.append(idx)
        #print X
        result = mds.fit_transform(np.asfarray(X,dtype='float'))
        x = []
        y = []
        c = []
        label = []
        for idx,v in enumerate(Y):
            label.append(v)
            x.append(result[idx][0])
            y.append(result[idx][1])
            c.append(pal[C[idx]])
    
        source = ColumnDataSource(data=dict(x=x,y=y, label=label))
        #Nodes Plotting
        o = figure(title="All Nodes",plot_height=self.ploth,plot_width=self.plotw)
        print "d"
        print c
        o.text('x', 'y', label, source=source, )
        o.circle('x', 'y', size=10, source=source,color=c )
        return o

    def nodes_figure(self):
        pal = pallete("nodes")
        palr = pallete("links")
        #mds = manifold.TSNE(n_components=2)
        mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
        X = []
        Y = []
        for v in self.m_points:
            X.append(self.m_points[v])
            Y.append(v)
        #print X
        result = mds.fit_transform(np.asfarray(X,dtype='float'))
        for idx,r in enumerate(Y):
            self.m_points[r] = result[idx] 
        x = []
        y = []
        d = []
        d1 = []
        d2 = []
        c=[]
        label = []
        nnodes = []
        for idx,v in enumerate(self.m_points):
            label.append(v)
            x.append(self.m_points[v][0])
            y.append(self.m_points[v][1])
            d.append(self.n_types_d[v])
            d1.append(self.n_types_d[v]*2*1.33333)
            d2.append(self.n_types_d[v]*2)
            nnodes.append(len(self.n_types[v]))
            c.append(pal[idx])
        source = ColumnDataSource(data=dict(x=x,y=y, label=label,d=d,nnodes=nnodes))
        #Nodes Plotting
        o = figure(title="Node Labels",plot_width=self.plotw,plot_height=self.ploth,tools="pan,wheel_zoom,box_zoom,reset,resize,hover")
        print "d"
        print d
        print d1
        print d2
        o.oval('x', 'y', width=d1,height=d2, source=source,alpha=0.7**len(self.m_points),height_units="data",width_units="data",color=c  )

        for idx,r in enumerate(self.r_types1):
            a1 = rotatePoint(self.m_points[self.r_types1[r]["s"]],self.m_points[self.r_types1[r]["t"]],np.degrees(self.r_types_d[idx])/2) 
            a2 = rotatePoint(self.m_points[self.r_types1[r]["s"]],self.m_points[self.r_types1[r]["t"]],-np.degrees(self.r_types_d[idx])/2) 
            o.segment(x0=self.m_points[self.r_types1[r]["s"]][0],x1=self.m_points[self.r_types1[r]["t"]][0],y0=self.m_points[self.r_types1[r]["s"]][1],y1=self.m_points[self.r_types1[r]["t"]][1],color=palr[idx],line_width=2.5)
            o.segment(x0=self.m_points[self.r_types1[r]["t"]][0],x1=a1[0],y0=self.m_points[self.r_types1[r]["t"]][1],y1=a1[1],color=palr[idx],line_width=2.5)
            o.segment(x0=self.m_points[self.r_types1[r]["t"]][0],x1=a2[0],y0=self.m_points[self.r_types1[r]["t"]][1],y1=a2[1],color=palr[idx],line_width=2.5)
            #o.segment(x0=a1[0],x1=a2[0],y0=a1[1],y1=a2[1])
            x = [self.m_points[self.r_types1[r]["s"]][0],a1[0],self.m_points[self.r_types1[r]["t"]][0],a2[0],self.m_points[self.r_types1[r]["s"]][0]]
            y = [self.m_points[self.r_types1[r]["s"]][1],a1[1],self.m_points[self.r_types1[r]["t"]][1],a2[1],self.m_points[self.r_types1[r]["s"]][1]]
            o.patch(x,y , alpha=0.3,color=palr[idx])

        o.text('x', 'y', label, source=source, )
        o.circle('x', 'y', size=10, source=source,fill_color=c)
        hover = o.select(dict(type=HoverTool))
        hover.tooltips = OrderedDict([
            ('Node Label', '@label'),
            ('Standard Deviation', '@d'),
            ('Number of Nodes', '@nnodes'),
        ])
        return o
    def links_figure(self):
        pal = pallete("links") 
        #Links Plotting
        q = figure(title="Link Labels")
        for idx,r in enumerate(self.r_types):
            x = np.linspace(math.pi+math.pi/2,3*math.pi+math.pi/2,2000)
            cdf = np.sin(x)*len(self.r_types[r])/2
            q.line(x*500000*self.r_types_d[idx] + idx*1000000, cdf+(len(self.r_types[r])/2), alpha=0.9,color=pal[idx],line_width=1)
            q.patch(x*500000*self.r_types_d[idx] + idx*1000000, cdf+(len(self.r_types[r])/2), alpha=0.7, legend=r,color=pal[idx])
        return q
    def angles_figure(self):
        #Links Matrix Plotting
        colormap = [
            "#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
            "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"
        ]
        xname = []
        yname = []
        color = []
        alpha = []
        angle = []
        names=[]
        for i, n1 in enumerate(self.r_types):
            names.append(n1)
            for j, n2 in enumerate(self.r_types):
                xname.append(n1)
                yname.append(n2)
                alpha.append(self.angle_matrix[n1][n2]/math.pi)
                angle.append(np.degrees(self.angle_matrix[n1][n2]))

                if n1 == n2:
                    color.append(colormap[2])
                else:
                    color.append('black')
        source = ColumnDataSource(
            
data=dict(
                xname=xname,
                yname=yname,
                colors=color,
                alphas=alpha,
                angles=angle
            )
        )
        p = figure(title="Angles Between Link labels",
            x_axis_location="above", tools="resize,hover,save",
            x_range=list(reversed(names)), y_range=names)
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
            ('names', '@yname, @xname'),
            ('angle', '@angles'),
        ])
        return p

    def similares(self,nodo,positives,negatives,tipo,label):
        my_list = self.w2v.most_similar(positives,negatives,topn=50000)
        result = []
        for m in my_list:
            if m[0] in self.n_types[tipo] and m[0] != nodo:
                result.append(m)
        return result

    def predice(self,nodo,label,tipo,rel,fast):
        if not fast:
            votos = []
            for r in self.r_types[rel]:            
                other = r["s"]
                if(r["s"] == nodo):
                    other = r["t"]
                p2 = neo4j.CypherQuery(self.graph_db, "match (n)-[:"+rel+"]-(m) where n."+label+' = "'+other+'" return m.'+label).execute()
                if len(p2) > 0:
                    for p in p2:
                        prop2 = p["m."+label]
                    prop2 = prop2.replace(" ","_")
                    other = other.replace(" ","_")
                    if other in self.w2v and prop2 in self.w2v:
                        prop1 = self.similares([nodo,other],[prop2],tipo,label)[0][0]
                        votos.append(prop1)
            return max(set(votos), key=votos.count)
        if fast:
            return self.similares(nodo,[self.w2v[nodo]+self.m_vectors[rel]],[],tipo,label)[0][0]

    def aciertos_rel(self,rel,label,fast):
        #print "ji"
        con = neo4j.CypherQuery(self.graph_db, "MATCH (a)-[r:"+rel+"]->(b) WHERE labels(a) <> [] AND labels(b) <> [] RETURN DISTINCT head(labels(a)) AS This, type(r) as To, head(labels(b)) AS That").execute()
        numaciertos = 0
        total = 0
        cuenta_misc = 0
        #print "ja"
        for rs in self.n_types[con[0]["This"]]:   
            cuenta_misc += 1
            print rs
            if rs in self.w2v and not '"' in rs and rs in self.sentences:
                #p2 = neo4j.CypherQuery(self.graph_db, "match (n)-[:"+rel+"]-(m) where n."+label+' = "'+r["s"]+'" return m.'+label).execute()
                #if r["s"] in self.sentences:
                total = total + 1
                print "tratando relacion "+str(cuenta_misc)+" de "+str(len(self.r_types[rel]))
                #print self.predice(rs,label,con[0]["That"],rel,fast)
                #print self.sentences[rs]
                #print numaciertos
                #print total
                if self.predice(rs,label,con[0]["That"],rel,fast) in self.sentences[rs]:#== p2[0]["m."+label]:
                    numaciertos += 1
        if total > 0:
            return float(numaciertos)/float(total)*100
        else:
            return 0

    def link_prediction_ratio(self):
        ratiosf = {}
        for r in self.r_types:
            if not os.path.exists("models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+self.mode+"-lpr-"+r+".p"):
                f = open( "models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+self.mode+"-lpr-"+r+".p", "w" )
                print "empezando cono tipo " +r+"..."
                ratiosf[r] = self.aciertos_rel(r,self.label,True)
                pickle.dump(ratiosf[r],f)
        else:
            f = open( "models/" + self.bd + str(self.ndim) +"d-"+str(self.ns)+"w"+str(self.w_size)+self.mode+"-lpr-"+r+".p", "r" )
            ratiosf[r] = pickle.load(f)

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
