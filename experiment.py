from node2vec import *
from sklearn import neighbors
import sys    # sys.setdefaultencoding is cancelled by site.py
reload(sys)    # to re-enable sys.setdefaultencoding()
sys.setdefaultencoding('utf-8')


class experiment:
    def __init__(self,bd,port,user,pss,label,mode,param,trainset_p):
        self.bd = bd 
        self.mode = mode 
        self.port = port
        self.user = user
        self.pss = pss
        self.label = label
        self.trainset_p = trainset_p
        self.param = param
        self.p = figure(plot_width=1000, plot_height=600)    
        self.ratiosf = {}
        self.r_desv = {}
        self.n_desv = {}

    def ntype_prediction(self,a,b,jump):
        pal = pallete("db")
        X = []
        Y = []


        for i in range(a,b+1):
            if self.param == "ns":
                k = 3
            if self.param == "l":
                k = 3
            if self.param == "ndim":
                k = 3
            if self.param == "k":
                k = i
            if not os.path.exists("models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)+".p"):
                if self.param == "ns":
                    n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,i*jump,200,6,self.mode,[])
                    k = 3
                if self.param == "l":
                    n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,1000000,200,i*jump,self.mode,[])
                    k = 3
                if self.param == "ndim":
                    n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,1000000,i*jump,6,self.mode,[])
                    k = 3
                if self.param == "k":
                    n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,1000000,200,6,self.mode,[])
                    k = i
                n2v.learn(self.mode,self.trainset_p)
                #k-neighbors for each node
                total = 0
                right = 0
                for t in n2v.n_types:
                    for n in  n2v.n_types[t]:
                        if n in n2v.w2v and random.random() < self.trainset_p:
                            votes = []
                            d = 10
                            while len(votes) < k:
                                votes = []
                                sim = n2v.w2v.most_similar(positive = [n],topn=d)
                                for s in sim:
                                    if n2v.ntype(s[0]) <> None:
                                        votes.append(n2v.ntype(s[0]))
                                d += 10          
                            if t == max(set(votes), key=votes.count):
                                right += 1
                            total += 1

                result = float(right)/float(total)
                f = open( "models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)+".p", "w" )
                pickle.dump(result,f)
            else:
                f = open( "models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)+".p", "r" )
                result = pickle.load(f)
            X.append(i*jump)
            Y.append(result)
        self.p.line(X, Y, color=pal[1],legend="ICH",line_width=1.5)
        self.p.legend.background_fill_alpha = 0.5
        return X,Y

    def ltype_prediction(self,a,b,jump):
        pal = pallete("db")
        X = []
        Y = []


        for i in range(a,b+1):
            if self.param == "ns":
                k = 3
            if self.param == "l":
                k = 3
            if self.param == "ndim":
                k = 3
            if self.param == "k":
                k = i
            if not os.path.exists("models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)+".p"):
                if self.param == "ns":
                    n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,i*jump,200,6,self.mode,[])
                    k = 3
                if self.param == "l":
                    n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,1000000,200,i*jump,self.mode,[])
                    k = 3
                if self.param == "ndim":
                    n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,1000000,i*jump,6,self.mode,[])
                    k = 3
                if self.param == "k":
                    n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,1000000,200,6,self.mode,[])
                    k = i
                n2v.learn(self.mode,self.trainset_p)
                #k-neighbors for each link
                total = 0
                right = 0
                X,Y =[],[]
                for t in n2v.r_types:
                    for r in n2v.r_types[t]:
                        X.append(r["v"])
                        Y.append(r)
                #n2v.connection.close()
                #n2v.db.close()
                #n2v.storage.close()

                clf = neighbors.KNeighborsClassifier(k, "uniform")
                clf.fit(X, Y)
                for t in n2v.r_types:
                    for r in n2v.r_types[t]:
                        if random.random() < self.trainset_p:
                            if clf.predict == r:
                                right += 1
                            total += 1
                result = float(right)/float(total)
                f = open( "models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)+".p", "w" )
                pickle.dump(result,f)
            else:
                f = open( "models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)+".p", "r" )
                result = pickle.load(f)
            X.append(i*jump)
            Y.append(result)
        self.p.line(X, Y, color=pal[1],legend="ICH",line_width=1.5)
        self.p.legend.background_fill_alpha = 0.5
        return X,Y



    def link_prediction(self,traversals,a,b,jump):
        pal = pallete("links")
        for i in range(a,b):
            if self.param == "ns":
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,i*jump,200,6,self.mode,traversals)
            if self.param == "l":
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,250000,200,i*jump,self.mode,traversals)
            if self.param == "ndim":
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,250000,i*jump,6,self.mode,traversals)
            n2v.delete_rels(self.trainset_p)
            n2v.learn(self.mode,self.trainset_p)
            for r in n2v.r_deleted:
                print r
                if r[0] == '-':
                    con = neo4j.CypherQuery(n2v.graph_db, "MATCH (a)"+r+"(b) WHERE labels(a) <> [] AND labels(b) <> [] RETURN DISTINCT head(labels(a)) AS This, type(r) as To, head(labels(b)) AS That LIMIT 5").execute()
                else:
                    con = neo4j.CypherQuery(n2v.graph_db, "MATCH (a)-[r:"+r+"]->(b) WHERE labels(a) <> [] AND labels(b) <> [] RETURN DISTINCT head(labels(a)) AS This, type(r) as To, head(labels(b)) AS That LIMIT 5").execute()
                if not r in self.ratiosf:
                    self.ratiosf[r] = []
                self.ratiosf[r].append(n2v.aciertos_rel(r,self.label,True,str(i*jump)+self.param+str(self.trainset_p)))
                if not r in self.r_desv:
                    self.r_desv[r] = []
                self.r_desv[r].append(n2v.r_desv[r])
                if not r in self.n_desv:
                    self.n_desv[r] = []
                print "MATCH (a)-[r:"+r+"]->(b) WHERE labels(a) <> [] AND labels(b) <> [] RETURN DISTINCT head(labels(a)) AS This, type(r) as To, head(labels(b)) AS That LIMIT 5"
                self.n_desv[r].append(n2v.n_types_d[con[0]["That"]])
                ndesv = []
        print self.ratiosf
        for j,r in enumerate(self.ratiosf):
            if r <> "HERRAMIENTA" and r <>  "PARROQUIA_LOC" and r <> "CANTON_LOC" and r <> "DSUBAMBITO_PERTENECE": 
                x = []
                ratios = []
                rdesv = []
                ndesv = []
                for i,ratio in enumerate(self.ratiosf[r]):
                    print "ja"
                    x.append(i*jump)
                    ratios.append(ratio)
                    rdesv.append(self.r_desv[r][i]*10)
                    ndesv.append(self.n_desv[r][i]*10)
                print ratios
                print x
                self.p.line(x, ratios, color=pal[j],legend=r+"(LPR)",line_width=1.5)
                self.p.line(x, rdesv, color=pal[j],legend=r+"(L-DESV)",line_dash='dashed')
                self.p.line(x, ndesv, color=pal[j],legend=r+"(N-DESV)",line_dash='dotted')
                self.p.legend.background_fill_alpha = 0.5

