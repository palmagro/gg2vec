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
            if self.param == "k":
                val = val - 1
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
                    n2v.learn(self.mode,self.trainset_p,False,it)
                    if self.param == "ns" or self.param == "ndim" or  self.param == "l":
                        result = predict("k",n2v.nodes_pos,n2v.nodes_type,val,self.trainset_p)
                    else:
                        result = predict(self.param,n2v.nodes_pos,n2v.nodes_type,val,self.trainset_p)
                    t += result
                    resultados.append(result)
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
        k = 3
        print "models/ntype_conf_matrix" + self.bd +"ts"+str(self.trainset_p)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p"
        if not os.path.exists("models/ntype_conf_matrix" + self.bd +"ts"+str(self.trainset_p)+"k"+str(k)+"Promedio"+str(self.iteraciones)+".p"):
            matrices = [None] * self.iteraciones
            #repetimos para self.iteraciones experimentos
            for it in range(self.iteraciones):
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,optimos[self.bd][0],optimos[self.bd][1],optimos[self.bd][2],self.mode,[],self.iteraciones)
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
                matriz_promedio[i][j] = round(float(matriz_promedio[i][j] * 100) / float(suma),2)
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
            if self.param == "k":
                val = val - 1
            if self.param == "ns":
                k = 3
            if self.param == "l":
                k = 3
            if self.param == "ndim":
                k = 3
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
                        result = predict("k",link_vectors,link_types,k,self.trainset_p)
                    else:
                        result = predict(self.param,link_vectors,link_types,val,self.trainset_p)
                    final += result
                    resultados.append(result)
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
            Yd.append(mean_dev*100)
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
                n2v = node2vec(self.bd,self.port,self.user,self.pss,self.label,optimos[self.bd][0],optimos[self.bd][1],optimos[self.bd][2],self.mode,[],self.iteraciones)
                n2v.learn(self.mode,self.trainset_p,False,it)
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
                if len(link_vectors) - 1 < k:
                    k1 = len(link_vectors) - 1
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
                matriz_promedio[i][j] = round(float(matriz_promedio[i][j] * 100) / float(suma),2)
        return matriz_promedio


    def link_prediction(self,traversals,a,b,jump,metrica,filtrado):
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
            if not os.path.exists("models/l_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p") or not os.path.exists("models/l_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p") or not os.path.exists("models/l_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p"):
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
                    n2v.learn(self.mode,self.trainset_p,True,it)
                    total = 0
                    parcial = 0
                    n2v.r_analysis()
                    if metrica == "MRR":
                        if filtrado:
                            clasificadores = {}
                            temp_pos = {}
                            temp_name = {}
                            ks = {}
                            total = 0
                            for rt in n2v.r_deleted:
                                temp_pos[rt] = []
                                temp_name[rt] = []
                                print "Se va a comparar con: " + str(n2v.r_deleted[rt][0]["tipot"])
                                for idx,e in enumerate(n2v.nodes_type):
                                    if e == n2v.r_deleted[rt][0]["tipot"]:
                                        temp_pos[rt].append(n2v.nodes_pos[idx])
                                        temp_name[rt].append(n2v.nodes_name[idx])
                                if len(temp_pos[rt]) < 1000:
                                    ks[rt] = len(temp_pos[rt])
                                else:
                                    ks[rt] = 1000                                
                                clasificadores[rt] = neighbors.KNeighborsClassifier(ks[rt], "uniform",n_jobs=multiprocessing.cpu_count())
                                clasificadores[rt].fit(temp_pos[rt], temp_name[rt])
                            print "A continuacion las aristas eliminadas"
                            for rt in n2v.r_deleted:
                                targettopredict = []
                                linkstopredictV = []
                                for d in n2v.r_deleted[rt]:
                                    rs = d["s"]
                                    rel = d["tipo"]
                                    tipot = d["tipot"]
                                    if rs in n2v.w2v and not '"' in rs:
                                        total += 1
                                        targettopredict.append(d["t"])
                                        linkstopredictV.append(n2v.w2v[rs]+n2v.m_vectors[str(rel)])
                                nbs = clasificadores[rt].kneighbors(linkstopredictV,ks[rt],False)
                                for idx,e in enumerate(nbs):
                                    nbs1 = []
                                    for i in e:
                                        nbs1.append(temp_name[rt][i])
                                    if targettopredict[idx] in nbs1:
                                        print "ESTA EN LA LISTA DEVUELTA"
                                        print targettopredict[idx]
                                        print nbs1.index(targettopredict[idx])
                                        parcial += float(1 / float(nbs1.index(targettopredict[idx])+1 ))
                                        print "PUNTUACION"
                                        print float(1 / float(nbs1.index(targettopredict[idx])+1 ))
                        else:
                            clf = neighbors.KNeighborsClassifier(1000, "uniform",n_jobs=multiprocessing.cpu_count())
                            clf.fit(n2v.nodes_pos, n2v.nodes_name)
                            print "A continuacion las aristas eliminadas"
                            targettopredict = []
                            linkstopredictV = []
                            for rt in n2v.r_deleted:
                                for d in n2v.r_deleted[rt]:
                                    rs = d["s"]
                                    rel = d["tipo"]
                                    tipot = d["tipot"]
                                    if rs in n2v.w2v and not '"' in rs:
                                        total += 1
                                        targettopredict.append(d["t"])
                                        linkstopredictV.append(n2v.w2v[rs]+n2v.m_vectors[str(rel)])
                            nbs = clf.kneighbors(linkstopredictV,1000,False)
                            for idx,e in enumerate(nbs):
                                nbs1 = []
                                for i in e:
                                    nbs1.append(n2v.nodes_name[i])
                                if targettopredict[idx] in nbs1:
                                    print "ESTA EN LA LISTA DEVUELTA"
                                    print targettopredict[idx]
                                    print nbs1.index(targettopredict[idx])
                                    parcial += float(1 / float(nbs1.index(targettopredict[idx])+1 ))
                                    print "PUNTUACION"
                                    print float(1 / float(nbs1.index(targettopredict[idx])+1 ))
                        if total > 0:
                            resultIN = float(parcial)/float(total)
                        else:
                            resultIN = 0
                    final += resultIN
                    resultados.append(resultIN)
                result = final / self.iteraciones                
                mean_dev = 0
                for r in resultados:
                    mean_dev += (r - result) * (r - result)
                mean_dev = math.sqrt(mean_dev)
                print "RESULTADOS DE UN PUNTO"
                print resultados
                print mean_dev
                f1 = open( "models/l_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "w" )
                pickle.dump(mean_dev,f1)
                f2 = open( "models/l_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "w" )
                pickle.dump(resultados,f2)
                f3 = open( "models/l_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "w" )
                pickle.dump(result,f3)
            else:
                f1 = open( "models/l_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "r" )
                mean_dev = pickle.load(f1)
                f2 = open( "models/l_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "r" )
                resultados = pickle.load(f2)
                f3 = open( "models/l_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "r" )
                result = pickle.load(f3)
            X.append(val)
            Y.append(result)
            Xd.append(val)
            Yd.append(mean_dev)
        self.p.line(X, Y, color=pal[1],legend="ICH",line_width=1.5)
        #self.p.line(Xd, Yd, color=pal[1],legend="ICH",line_width=1.5,line_dash='dotted')
        self.p.legend.background_fill_alpha = 0.5
        print self.bd
        print "max accuracy: " + str(max(Y))
        print "max dev: " + str(max(Yd))
        return X,Y,Xd,Yd

    def traversal_prediction(self,traversal,a,b,jump,metrica,filtrado):
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
            if not os.path.exists("models/"+traversal+"_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p") or not os.path.exists("models/"+traversal+"_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p") or not os.path.exists("models/"+traversal+"_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p"):
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
                    n2v.learn(self.mode,0,False,it)
                    parcial = 0
                    n2v.r_analysis()
                    #Obtenemos el vector medio del traversal solicitado. 
                    v_traversal = n2v.get_vtraversal(traversal)
                    #Obtenemos una serie de traversals por los que vamos a preguntar. Es una lista que contiene diccionarios con: nodo origen (s), nodo destino (t) y tipo del nodo destino (tipot).
                    traversals = n2v.get_traversals(traversal,self.trainset_p)
                    if metrica == "MRR":
                        if filtrado:
                            temp_pos = []
                            temp_name = []
                            print "Se va a comparar con: " + str(traversals[0]["tipot"])
                            for idx,e in enumerate(n2v.nodes_type):
                                if e == traversals[0]["tipot"]:
                                    temp_pos.append(n2v.nodes_pos[idx])
                                    temp_name.append(n2v.nodes_name[idx])
                            if len(temp_pos) < 1000:
                                ks = len(temp_pos)
                            else:
                                ks = 1000                                
                            clasificador = neighbors.KNeighborsClassifier(ks, "uniform",n_jobs=multiprocessing.cpu_count())
                            clasificador.fit(temp_pos, temp_name)
                            print "A continuacion la verificacion de traversals"
                            targettopredict = []
                            linkstopredictV = []
                            for t in traversals:
                                rs = t["s"]
                                tipot = t["tipot"]
                                if rs in n2v.w2v and not '"' in rs:
                                    targettopredict.append(t["t"])
                                    linkstopredictV.append(n2v.w2v[rs]+v_traversal)
                            print "Tamanio del conjunto de entrenamiento"
                            print len(linkstopredictV)
                            total = len(linkstopredictV)
                            nbs = clasificador.kneighbors(linkstopredictV,ks,False)
                            for idx,e in enumerate(nbs):
                                nbs1 = []
                                for i in e:
                                    nbs1.append(temp_name[i])
                                if targettopredict[idx] in nbs1:
                                    print "ESTA EN LA LISTA DEVUELTA"
                                    print targettopredict[idx]
                                    print nbs1.index(targettopredict[idx])
                                    parcial += float(1 / float(nbs1.index(targettopredict[idx])+1 ))
                                    print "PUNTUACION"
                                    print float(1 / float(nbs1.index(targettopredict[idx])+1 ))
                        else:
                            clf = neighbors.KNeighborsClassifier(1000, "uniform",n_jobs=multiprocessing.cpu_count())
                            clf.fit(n2v.nodes_pos, n2v.nodes_name)
                            print "A continuacion la verificacion de traversals"
                            targettopredict = []
                            linkstopredictV = []
                            for t in traversals:
                                rs = t["s"]
                                tipot = t["tipot"]
                                if rs in n2v.w2v and not '"' in rs:
                                    targettopredict.append(t["t"])
                                    linkstopredictV.append(n2v.w2v[rs]+v_traversal)
                            total = len(linkstopredictV)
                            nbs = clf.kneighbors(linkstopredictV,1000,False)
                            for idx,e in enumerate(nbs):
                                nbs1 = []
                                for i in e:
                                    nbs1.append(n2v.nodes_name[i])
                                if targettopredict[idx] in nbs1:
                                    print "ESTA EN LA LISTA DEVUELTA"
                                    print targettopredict[idx]
                                    print nbs1.index(targettopredict[idx])
                                    parcial += float(1 / float(nbs1.index(targettopredict[idx])+1 ))
                                    print "PUNTUACION"
                                    print float(1 / float(nbs1.index(targettopredict[idx])+1 ))
                        if total > 0:
                            print parcial
                            print total
                            resultIN = float(parcial)/float(total)
                            print resultIN
                        else:
                            resultIN = 0
                    final += resultIN
                    resultados.append(resultIN)
                result = final / self.iteraciones                
                mean_dev = 0
                for r in resultados:
                    mean_dev += (r - result) * (r - result)
                mean_dev = math.sqrt(mean_dev)
                print "RESULTADOS DE UN PUNTO"
                print resultados
                print mean_dev
                f1 = open( "models/"+traversal+"_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "w" )
                pickle.dump(mean_dev,f1)
                f2 = open( "models/"+traversal+"_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "w" )
                pickle.dump(resultados,f2)
                f3 = open( "models/"+traversal+"_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "w" )
                pickle.dump(result,f3)
            else:
                f1 = open( "models/"+traversal+"_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"MeanDev"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "r" )
                mean_dev = pickle.load(f1)
                f2 = open( "models/"+traversal+"_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Resultados"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "r" )
                resultados = pickle.load(f2)
                f3 = open( "models/"+traversal+"_prediction" + self.bd +"ts"+str(self.trainset_p)+self.param+str(val)+"k"+str(k)+"Promedio"+"Metrica-"+str(metrica)+"Filtrado-"+str(filtrado)+str(self.iteraciones)+".p", "r" )
                result = pickle.load(f3)
            X.append(val)
            Y.append(result*100)
            Xd.append(val)
            Yd.append(mean_dev*100)
        self.p.line(X, Y, color=pal[1],legend="ICH",line_width=1.5)
        #self.p.line(Xd, Yd, color=pal[1],legend="ICH",line_width=1.5,line_dash='dotted')
        self.p.legend.background_fill_alpha = 0.5
        print self.bd
        print "YOO"
        print "max accuracy: " + str(max(Y))
        print "max dev: " + str(max(Yd))
        return X,Y,Xd,Yd
