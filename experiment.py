import pydot 
from node2vec import *
from sklearn import neighbors,metrics
from credentials import *
import sys    # sys.setdefaultencoding is cancelled by site.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
reload(sys)    # to re-enable sys.setdefaultencoding()
sys.setdefaultencoding('utf-8')
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors.kde import KernelDensity
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#el valor de trainset_p sera usado como probabilidad de que un elemento sea evaluado con knn (valor entre 0 y 1 ) o como valor de la cantidad de folds en el cross valdiation para cualquier de los otros metodos.
class experiment:
    def __init__(self,bd,port,user,pss,label,mode,param,trainset_p,iteraciones):
        self.bd = bd 
        self.mode = mode 
        self.port = port
        self.user = user
        self.pss = pss
        self.label = label
        self.trainset_p = trainset_p
        self.param = param
        self.p = figure(plot_width=600, plot_height=400)    
        self.ratiosf = {}
        self.r_desv = {}
        self.n_desv = {}
        self.iteraciones = iteraciones

    def ntype_prediction(self,a,b,jump):
        pal = pallete("db")
        # Valores para la grafica de precision en la prediccion
        X = []
        Y = []
        # Valores para la grafica de desviacion en la prediccion
        Xd = []
        Yd = []
        i = 1
        for i in range(a,b+1):
            val = i * jump  
            if self.param == "ns":
                k = 3
            if self.param == "l":
                k = 3
            if self.param == "ndim":
                k = 3
            if not (self.param == "ns" or self.param == "ndim" or self.param == "l"):
                k = val
            resultados = []     
            print "models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p"
            print os.path.exists("models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p")   
            if not os.path.exists("models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p") or not os.path.exists("models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+str(self.iteraciones)+".p") or not os.path.exists("models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+str(self.iteraciones)+".p"):
                t = 0
                for it in range(self.iteraciones):
                    if self.param == "ns":
                        n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,val,200,6,self.mode,[],self.iteraciones)
                        k = 3
                    if self.param == "l":
                        n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,400000,200,val,self.mode,[],self.iteraciones)
                        k = 3
                    if self.param == "ndim":
                        n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,400000,val,6,self.mode,[],self.iteraciones)
                        k = 3
                    #si lo que vamos a estudiar no son los parametros libres de la inmersion, fijamos dichos parametros a sus valores optimos segun BD
                    if not (self.param == "ns" or self.param == "ndim" or self.param == "l"):
                        n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,optimos[self.bd][0],optimos[self.bd][1],optimos[self.bd][2],self.mode,[],self.iteraciones)
                    n2v.connectZODB()
                    n2v.learn(self.mode,self.trainset_p,False,it)
                    if self.param == "ns" or self.param == "ndim" or  self.param == "l":
                        result = predict("k",n2v.nodes_pos,n2v.nodes_type,val,self.trainset_p)
                    else:
                        result = predict(self.param,n2v.nodes_pos,n2v.nodes_type,val,self.trainset_p)
                    t += result
                    resultados.append(result)
                    n2v.disconnectZODB()
                    print result
                result = t / self.iteraciones
                mean_dev = 0
                for r in resultados:
                    mean_dev += (r - result) * (r - result)
                mean_dev = math.sqrt(mean_dev)
                f1 = open( "models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+str(self.iteraciones)+".p", "w" )
                pickle.dump(mean_dev,f1)
                f2 = open( "models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+str(self.iteraciones)+".p", "w" )
                pickle.dump(resultados,f2)
                f3 = open( "models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p", "w" )
                pickle.dump(result,f3)
            else:
                f1 = open( "models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+str(self.iteraciones)+".p", "r" )
                mean_dev = pickle.load(f1)
                f2 = open( "models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+str(self.iteraciones)+".p", "r" )
                resultados = pickle.load(f2)
                f3 = open( "models/ntype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p", "r" )
                result = pickle.load(f3)
            #print "RESULT"
            #print result
            #print "RESULTADOS"
            #print resultados
            #print "MEAN DEV"
            #print mean_dev
            X.append(val)
            Y.append(result*100)
            Xd.append(val)
            Yd.append(mean_dev)
        print self.bd
        print "max accuracy: " + str(max(Y))
        print "max dev: " + str(max(Yd))
        self.p.line(X, Y, color=pal[1],legend=self.bd,line_width=1.5)
        #self.p.line(Xd, Yd, color=pal[1],legend=self.bd + " dev",line_width=1.5,line_dash='dotted')
        self.p.legend.background_fill_alpha = 0.5
        return X,Y,Xd,Yd
    
    def ntype_conf_matrix(self):
        k = 100000
        if not os.path.exists("models/ntype_conf_matrix" + self.bd +"ts"+str(self.trainset_p)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p") or True:
            matrices = [None] * self.iteraciones
            #repetimos para self.iteraciones experimentos
            for it in range(self.iteraciones):
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,800000,200,6,self.mode,[],self.iteraciones)
                n2v.connectZODB()
                n2v.learn(self.mode,self.trainset_p,False,it)
                #generamos un diccionario para saber las posiciones de cada tipo de nodo en la matriz
                dic = dict()
                for idx,t in enumerate(n2v.n_types):
                    dic[t] = idx
                #generamos la matriz para cada experimento
                matriz = [0] * (len(n2v.n_types)+1)
                for i in range(0,len(n2v.n_types)+1):
                    matriz[i] = [0] * (len(n2v.n_types)+1)
                    for idx,t in enumerate(n2v.n_types):
                        if i == 0:
                            matriz[i][idx+1] = t
                        else:
                            matriz[i][idx] = 0    
                for idx,t in enumerate(n2v.n_types):
                    matriz[idx+1][0] = t
                #k-neighbors for each node
                pos = []
                types = []
                for idx,i in enumerate(n2v.nodes_pos):
                    if random.random() < self.trainset_p:
                        pos.append(i)
                        types.append(n2v.nodes_type[idx])
                if len(pos) - 1 < k:
                    k1 = len(pos) - 1
                else:
                    k1 = k
                clf = neighbors.KNeighborsClassifier(k1+1, "uniform",n_jobs=multiprocessing.cpu_count())
                clf.fit(n2v.nodes_pos, n2v.nodes_type)
                neigh = clf.kneighbors(pos,return_distance = False)
                for idx,n in enumerate(neigh):
                    votes = []                    
                    for idx1,s in enumerate(neigh[idx][1:]):
                        votes.append(n2v.nodes_type[s])
                    matriz[dic[types[idx]]+1][dic[max(set(votes), key=votes.count)]+1] +=1
                n2v.disconnectZODB()
                print matriz
                matrices[it] = matriz
            f = open( "models/ntype_conf_matrix" + self.bd +"ts"+str(self.trainset_p)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p", "w" )
            pickle.dump(matrices,f)
        else:
            f = open( "models/ntype_conf_matrix" + self.bd +"ts"+str(self.trainset_p)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p", "r" )
            matrices = pickle.load(f)
        #calculando la matriz de confusion promedios de n experimentos
        matriz_promedio = [None] * (len(matrices[0]))
        for i in range(0,len(matrices[0])):
            matriz_promedio[i] = [0] * (len(matrices[0]))
        for idx,t in enumerate(matrices[0]):
            matriz_promedio[0][idx] = t[0]
        for idx,t in enumerate(matrices[0]):
            matriz_promedio[idx][0] = t[0]
        for i in range(1,len(matrices[0])):
            for j in range(1,len(matrices[0])):
                suma = 0
                for m in range(self.iteraciones):
                    suma += matrices[m][i][j]
                matriz_promedio[i][j] = float(suma)/float(self.iteraciones)
        #calculando porcentajes a partir del promedio de frecuencias
        for i in range(1,len(matriz_promedio)):
            suma = 0
            for j in range(1,len(matriz_promedio)):                
                suma += matriz_promedio[i][j]
            for j in range(1,len(matriz_promedio)):                
                matriz_promedio[i][j] = str(round(float(matriz_promedio[i][j] * 100) / float(suma),2))+"%"
        #Poniendo negritas
        for i in range(1,len(matriz_promedio)):
            matriz_promedio[i][0] =  "\meg{ "+str(matriz_promedio[i][0])+"}"
            matriz_promedio[0][i] =  "\meg{ "+matriz_promedio[0][i]+"}"
        for i in range(1,len(matriz_promedio)):
            matriz_promedio[i][i] =  "\meg{ "+matriz_promedio[i][i]+"}"
        matriz_promedio[0][0] = ""
        return matriz_promedio


    #Por ahora esta preparado para recibir solo dos tipos que se solapan!
    def nmultitype_conf_matrix(self,tipos,nfolds):
        cadena = ""
        for t in tipos:
            cadena += t
        if not os.path.exists("models/nmultitype_conf_matrix" + self.bd +"ts"+cadena+"Promedio"+str(nfolds)+".p") or True:
            #Creamos la matriz de matrices donde guardaremos los resultados parciales
            matrices = [None] * nfolds * nfolds
            #Creamos/Recuperamos el modelo Node2Vec
            n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,1000,20,6,self.mode,[],1)
            n2v.learn("normal",0,False,0)
            #Creamos los arrays X e Y, anadiendo
            X = []
            Y = []
            #Creamos un array de comunes que son los nodos que son a la vez de ambos tipos
            comunes = list()
            for tipo in tipos:
                for n in n2v.n_types[tipo]:
                    if n in n2v.w2v:
                        X.append(n2v.w2v[n])
                        if n in n2v.n_types[tipos[0]] and  n in n2v.n_types[tipos[1]]:
                            comunes.append(n2v.w2v[n])
                        Y.append(tipo)
            #Creamos los k folds estratificados    
            X = np.array(X)
            Y = np.array(Y)
            skf = StratifiedKFold(Y, n_folds=nfolds)
            it = 0
            kdes = []
            for train_index, test_index in skf:
                print "k-fold para kde"
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                Y_test = Y_test.astype('|S64')
                #Creamos la funcion de densidad de probabilidad de cada tipo
                for t in tipos:
                    print "Creando KDE para el tipo "+t
                    tempX = []
                    for idx,n in enumerate(Y_train):
                        if n == t:
                            tempX.append(X_train[idx])
                    #Calculating KDE with the train set
                    #use grid search cross-validation to optimize the bandwidth
                    #params = {'bandwidth': np.logspace(-1, 1, 10)}
                    #grid = GridSearchCV(neighbors.KernelDensity(), params)
                    #grid.fit(tempX)
                    #print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
                    # use the best estimator to compute the kernel density estimate
                    #kde = grid.best_estimator_
                    kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
                    kde.fit(tempX)
                    kdes.append(kde)
                    print "Terminado KDE para el tipo "+t
                #Dividimos el conjunto de test en tipo1, tipo2 y tipo1+2
                cont = 0
                for idx,x in enumerate(X_test):
                    total = 0
                    x = np.array(x)
                    if any((x == a).all() for a in comunes):
                        Y_test[idx] = str(tipos[0]+"+"+tipos[1])
                        cont += 1
                print "Numero de elementos con doble tipo:"+str(cont)
                #Creamos k-folds estratificados para el arbol de decision
                skf = StratifiedKFold(Y_test, n_folds=nfolds)
                for train_index, test_index in skf:
                    print "k-fold para decission tree"
                    X_train1, X_test1 = X_test[train_index], X_test[test_index]
                    Y_train1, Y_test1 = Y_test[train_index], Y_test[test_index]
                    clf = DecisionTreeClassifier(random_state=0)
                    print X_train1[0]
                    clf.fit(X_train1,Y_train1)
                    export_graphviz(clf);
                    Y_pred1 = clf.predict(X_test1)
                    matriz = metrics.confusion_matrix(Y_test1, Y_pred1,[tipos[0],tipos[1],tipos[0]+"+"+tipos[1]])
                    matrices[it] = np.array(matriz)
                    print matrices[it]
                    it += 1
            f = open( "models/nmultitype_conf_matrix" + self.bd +"ts"+cadena+"Promedio"+str(nfolds)+".p", "w" )
            pickle.dump(matrices,f)
        else:
            f = open( "models/nmultitype_conf_matrix" + self.bd +"ts"+cadena+"Promedio"+str(nfolds)+".p", "r" )
            matrices = pickle.load(f)
        total = matrices[0]
        for m in matrices[1:]:
            total += m
        print total
        matriz_promedio = total 
        matriz_promedio = matriz_promedio.astype('float')
        #print matrices
        #print matriz_promedio
        matriz_promedio = matriz_promedio / len(matrices)
        #print matriz_promedio
        #calculando porcentajes a partir del promedio de frecuencias
        for i in range(0,len(matriz_promedio)):
            suma = 0
            for j in range(0,len(matriz_promedio)): 
                suma += matriz_promedio[i][j]
                matriz_promedio[i][j] = float(matriz_promedio[i][j])
            for j in range(0,len(matriz_promedio)):                
                if suma > 0:
                    matriz_promedio[i][j] = round(float(matriz_promedio[i][j] * 100) / float(suma),2)
                else:
                    matriz_promedio[i][j] = 0
        matriz_promedio = matriz_promedio.astype('string')
        for i in range(0,len(matriz_promedio)):
            for j in range(0,len(matriz_promedio)):
                matriz_promedio[i][j] = str(matriz_promedio[i][j])+"%"
        return matriz_promedio



    def ltype_prediction(self,a,b,jump):
        # Valores para la grafica de precision en la prediccion
        pal = pallete("db")
        X = []
        Y = []
        # Valores para la grafica de desviacion en la prediccion
        Xd = []
        Yd = []
        i = 1
        for i in range(a,b+1):
            val = i * jump    
            if self.param == "ns":
                k = 3
            if self.param == "l":
                k = 3
            if self.param == "ndim":
                k = 3
            if not (self.param == "ns" or self.param == "ndim" or self.param == "l"):
                k = val
            resultados = []                
            if not os.path.exists("models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p") or not os.path.exists("models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+str(self.iteraciones)+".p") or not os.path.exists("models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+str(self.iteraciones)+".p"):
                final = 0
                for it in range(self.iteraciones):
                    if self.param == "ns":
                        n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,val,200,6,self.mode,[],self.iteraciones)
                        k = 3
                    if self.param == "l":
                        n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,400000,200,val,self.mode,[],self.iteraciones)
                        k = 3
                    if self.param == "ndim":
                        n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,400000,val,6,self.mode,[],self.iteraciones)
                        k = 3
                    #si lo que vamos a estudiar no son los parametros libres de la inmersion, fijamos dichos parametros a sus valores optimos segun BD
                    if not (self.param == "ns" or self.param == "ndim" or self.param == "l"):
                        n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,optimos[self.bd][0],optimos[self.bd][1],optimos[self.bd][2],self.mode,[],self.iteraciones)
                    n2v.connectZODB()
                    n2v.learn(self.mode,self.trainset_p,False,it)
                    #k-neighbors for each node
                    total = 0
                    right = 0
                    link_vectors = []
                    link_types = []
                    for t in n2v.r_types:
                        for r in n2v.r_types[t]:
                            link_vectors.append(r["v"])
                            link_types.append(t)
                    if self.param == "ns" or self.param == "ndim" or  self.param == "l":
                        result = predict("k",link_vectors,link_types,val,self.trainset_p)
                    else:
                        result = predict(self.param,link_vectors,link_types,val,self.trainset_p)
                    final += result
                    resultados.append(result)
                    n2v.disconnectZODB()
                result = final / self.iteraciones                
                mean_dev = 0
                for r in resultados:
                    mean_dev += (r - result) * (r - result)
                mean_dev = math.sqrt(mean_dev)
                f1 = open( "models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+str(self.iteraciones)+".p", "w" )
                pickle.dump(mean_dev,f1)
                f2 = open( "models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+str(self.iteraciones)+".p", "w" )
                pickle.dump(resultados,f2)
                f3 = open( "models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p", "w" )
                pickle.dump(result,f3)
            else:
                f1 = open( "models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+str(self.iteraciones)+".p", "r" )
                mean_dev = pickle.load(f1)
                f2 = open( "models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+str(self.iteraciones)+".p", "r" )
                resultados = pickle.load(f2)
                f3 = open( "models/ltype_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p", "r" )
                result = pickle.load(f3)
            X.append(val)
            Y.append(result*100)
            Xd.append(val)
            Yd.append(mean_dev)
        self.p.line(X, Y, color=pal[1],legend="ICH",line_width=1.5)
        #self.p.line(Xd, Yd, color=pal[1],legend="ICH",line_width=1.5,line_dash='dotted')
        self.p.legend.background_fill_alpha = 0.5
        print self.bd
        print "max accuracy: " + str(max(Y))
        print "max dev: " + str(max(Yd))
        return X,Y,Xd,Yd

    def ltype_conf_matrix(self):
        k = 3
        if not os.path.exists("models/ltype_conf_matrix" + self.bd +"ts"+str(self.trainset_p)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p"):
            matrices = [None] * self.iteraciones
            #repetimos para self.iteraciones experimentos
            for it in range(self.iteraciones):
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,400000,200,6,self.mode,[],self.iteraciones)
                n2v.connectZODB()
                n2v.learn(self.mode,self.trainset_p,False)
                #generamos un diccionario para saber las posiciones de cada tipo de nodo en la matriz
                dic = dict()
                for idx,t in enumerate(n2v.r_types):
                    dic[t] = idx
                #generamos la matriz para cada experimento
                matriz = [0] * (len(n2v.r_types)+1)
                for i in range(0,len(n2v.r_types)+1):
                    matriz[i] = [0] * (len(n2v.r_types)+1)
                    for idx,t in enumerate(n2v.r_types):
                        if i == 0:
                            matriz[i][idx+1] = t
                        else:
                            matriz[i][idx] = 0    
                for idx,t in enumerate(n2v.r_types):
                    matriz[idx+1][0] = t
                #k-neighbors for each node
                link_vectors = []
                link_types = []
                for t in n2v.r_types:
                    for r in n2v.r_types[t]:
                        link_vectors.append(r["v"])
                        link_types.append(t)
                if len(pos) - 1 < k:
                    k1 = len(pos) - 1
                else:
                    k1 = k
                clf = neighbors.KNeighborsClassifier(k1+1, "uniform",n_jobs=multiprocessing.cpu_count())
                clf.fit(link_vectors, link_types)
                pos = []
                types = []
                for idx,i in enumerate(link_vectors):
                    if random.random() < self.trainset_p:
                        pos.append(i)
                        types.append(link_types[idx])
                neigh = clf.kneighbors(pos,return_distance = False)
                for idx,n in enumerate(neigh):
                    votes = []                    
                    for idx1,s in enumerate(neigh[idx][1:]):
                        votes.append(link_types[s])
                    matriz[dic[types[idx]]+1][dic[max(set(votes), key=votes.count)]+1] +=1
                n2v.disconnectZODB()
                print matriz
                matrices[it] = matriz
            f = open( "models/ltype_conf_matrix" + self.bd +"ts"+str(self.trainset_p)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p", "w" )
            pickle.dump(matrices,f)
        else:
            f = open( "models/ltype_conf_matrix" + self.bd +"ts"+str(self.trainset_p)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p", "r" )
            matrices = pickle.load(f)
        #calculando la matriz de confusion promedios de n experimentos
        matriz_promedio = [None] * (len(matrices[0]))
        for i in range(0,len(matrices[0])):
            matriz_promedio[i] = [0] * (len(matrices[0]))
        for idx,t in enumerate(matrices[0]):
            matriz_promedio[0][idx] = t[0]
        for idx,t in enumerate(matrices[0]):
            matriz_promedio[idx][0] = t[0]
        for i in range(1,len(matrices[0])):
            for j in range(1,len(matrices[0])):
                suma = 0
                for m in range(self.iteraciones):
                    suma += matrices[m][i][j]
                matriz_promedio[i][j] = float(suma)/float(self.iteraciones)
        #calculando porcentajes a partir del promedio de frecuencias
        for i in range(1,len(matriz_promedio)):
            suma = 0
            for j in range(1,len(matriz_promedio)):                
                suma += matriz_promedio[i][j]
            for j in range(1,len(matriz_promedio)):                
                matriz_promedio[i][j] = str(round(float(matriz_promedio[i][j] * 100) / float(suma),2))+"%"
        #Poniendo negritas
        for i in range(1,len(matriz_promedio)):
            matriz_promedio[i][0] =  "\meg{ "+str(matriz_promedio[i][0])+"}"
            matriz_promedio[0][i] =  "\meg{ "+matriz_promedio[0][i]+"}"
        for i in range(1,len(matriz_promedio)):
            matriz_promedio[i][i] =  "\meg{ "+matriz_promedio[i][i]+"}"
        matriz_promedio[0][0] = ""
        return matriz_promedio


    def link_prediction(self,traversals,a,b,jump):
        pal = pallete("links")
        self.ratiosf = {}
        self.n_desv = {}
        self.r_desv = {}
        for i in range(a,b):

            if self.param == "ns":
                k = 3
            if self.param == "l":
                k = 3
            if self.param == "ndim":
                k = 3
            if self.param == "k":
                k = i
            if self.param == "ns":
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,i*jump,200,6,self.mode,traversals)
            if self.param == "l":
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,250000,200,i*jump,self.mode,traversals)
            if self.param == "ndim":
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,250000,i*jump,6,self.mode,traversals)
            n2v.connectZODB()
            if "lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k) not in n2v.root:
                print "entrando"
                n2v.learn(self.mode,self.trainset_p,False)
                n2v.delete_rels(self.trainset_p)
                n2v.learn(self.mode,self.trainset_p,True)
                n2v.n_analysis()
                n2v.r_analysis()

                n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)] = PersistentDict()
                n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["ratiosf"] = PersistentDict()
                n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["r_desv"] = PersistentDict()
                n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["n_desv"] = PersistentDict()
                total = 0
                suma = 0
                for r in n2v.r_deleted:
                    print r
                    if r[0] == '-':
                        con = neo4j.CypherQuery(n2v.graph_db, "MATCH (a)"+r+"(b) WHERE labels(a) <> [] AND labels(b) <> [] RETURN DISTINCT head(labels(a)) AS This, type(r) as To, head(labels(b)) AS That LIMIT 5").execute()
                    else:
                        con = neo4j.CypherQuery(n2v.graph_db, "MATCH (a)-[r:"+r+"]->(b) WHERE labels(a) <> [] AND labels(b) <> [] RETURN DISTINCT head(labels(a)) AS This, type(r) as To, head(labels(b)) AS That LIMIT 5").execute()
                    if not r in self.ratiosf:
                        self.ratiosf[r] = []
                    aciertos = n2v.aciertos_rel(r,self.label,True,str(i*jump)+self.param+str(self.trainset_p))
                    self.ratiosf[r].append(aciertos)
                    n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["ratiosf"][r] = aciertos
                    if not r in self.r_desv:
                        self.r_desv[r] = []
                    self.r_desv[r].append(n2v.r_desv[r])
                    n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["r_desv"][r] = n2v.r_desv[r]
                    if not r in self.n_desv:
                        self.n_desv[r] = []
                    print "MATCH (a)-[r:"+r+"]->(b) WHERE labels(a) <> [] AND labels(b) <> [] RETURN DISTINCT head(labels(a)) AS This, type(r) as To, head(labels(b)) AS That LIMIT 5"
                    self.n_desv[r].append(n2v.n_types_d[con[0]["That"]])
                    n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["n_desv"][r]  = n2v.n_types_d[con[0]["That"]]
                    ndesv = []
            else:
                for r in n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["ratiosf"]:
                    if not r in self.ratiosf:
                        self.ratiosf[r] = []
                    self.ratiosf[r].append(n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["ratiosf"][r])    
                    if not r in self.r_desv:
                        self.r_desv[r] = []
                    self.r_desv[r].append(n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["r_desv"][r])  
                    if not r in self.n_desv:
                        self.n_desv[r] = []  
                    self.n_desv[r].append(n2v.root["lp" + self.bd +"ts"+str(self.trainset_p)+self.param+str(i*jump)+"k"+str(k)]["n_desv"][r])    
            n2v.disconnectZODB()
        print self.ratiosf
        print self.r_desv
        print self.n_desv
        
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
                    rdesv.append(self.r_desv[r][i])
                    ndesv.append(self.n_desv[r][i])
                print ratios
                print x
                self.p.line(x, ratios, color=pal[j],legend=r+"(LPR)",line_width=1.5)
                self.p.line(x, rdesv, color=pal[j],legend=r+"(L-DESV)",line_dash='dashed')
                self.p.line(x, ndesv, color=pal[j],legend=r+"(N-DESV)",line_dash='dotted')
                self.p.legend.background_fill_alpha = 0.5

