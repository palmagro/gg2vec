import numpy as np
from math import acos
from sklearn import neighbors,metrics
import multiprocessing
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm,cross_validation
from sknn.mlp import Classifier,Layer
from sklearn.cross_validation import StratifiedKFold
#Diccionario que almacena los valores optimos de la inmersion para cada una de las bases de datos
optimos = {"cine":[400000,150,3],"inmaterial":[300000,20,2],"wordnet":[1000000,50,8]}

def dotproduct(a,b):
	return sum([a[i]*b[i] for i in range(len(a))])

#Calculates the size of a vector
def veclength(a):
#    print sum([a[i] for i in range(len(a))]) ** .5  
    return np.linalg.norm(a)

#Calculates the angle between two vector
def angle(a,b):
    dp=dotproduct(a,b)
    la=veclength(a)
    lb=veclength(b)
    costheta=min(1,dp/(la*lb))
    costheta=max(-1,costheta)
    return acos(costheta)

def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack 
    result = [None] * k
    for i in xrange(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result

def rotatePoint(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point

def generate_sample(mode,sentences_array,degree,w_size,i):
    if mode == "degree":
        s = sentences_array[weighted_choice(degree)]
    else:
        s = np.random.choice(sentences_array)
    s = eval(str(s))               
    a = s[0] 
    b = sample_wr(s[1:],w_size)
    b.insert(0,a)
    return b

def weighted_choice(weights):
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i

#Funcion que recibe un tipo de modelo de prediccion, las posiciones de los nodos/aristas, sus tipos y el valor del "parametro libre" del modelo pasado como primer parametro (el parametro ts: training set solo se usa para k vecinos, porque en esa epoca no usaba validacion cruzada)
def predict(model, pos, types, val,ts):
    if model == "k":
        k = val
        #k-neighbors for each node
        total = 0
        right = 0
        pos1 = []
        types1 = []
        for idx,i in enumerate(pos):
            if random.random() < ts:
                pos1.append(i)
                types1.append(types[idx])
        if len(pos) - 1 < k:
            k1 = len(pos) - 1
        else:
            k1 = k
        clf = neighbors.KNeighborsClassifier(k1+1, "uniform",n_jobs=multiprocessing.cpu_count())
        print len(pos)
        print len(types)
        clf.fit(pos, types)
        neigh = clf.kneighbors(pos1,return_distance = False)
        for idx,n in enumerate(neigh):
            votes = []                    
            for idx1,s in enumerate(neigh[idx][1:]):
                votes.append(types[s])
            if types1[idx] == max(set(votes), key=votes.count):
                right += 1
            total += 1
        return float(right)/float(total)
    if model == "SVM":
        #C Support Vector Clasification
        clf = svm.SVC(kernel='linear', C=val)
        scores = cross_validation.cross_val_score(clf, pos, types, cv=ts)
        print scores.mean()
        return scores.mean()
    if model == "RF":
        #C Support Vector Clasification
        clf = RandomForestClassifier(n_estimators = val)
        scores = cross_validation.cross_val_score(clf, pos, types, cv=ts)
        return scores.mean()
    if model == "ANN":
        #C Support Vector Clasification
        clf = Classifier(layers=[Layer("Sigmoid", units=val),Layer("Softmax")])
        skf = StratifiedKFold(types, n_folds=ts)
        it = 0
        kdes = []
        scores = []
        for train_index, test_index in skf:
            scores.append(0)
            print "k-fold para ANN"
            X_train, X_test = list( pos[i] for i in train_index ), list( pos[i] for i in test_index )
            Y_train, Y_test = list( types[i] for i in train_index ), list( types[i] for i in test_index )
            X_train, X_test = np.array(X_train), np.array(X_test)
            Y_train, Y_test = np.array(Y_train), np.array(Y_test)
            clf.fit(X_train, Y_train)
            Y_predicted = clf.predict(X_test)
            right = 0
            total = 0
            for idx,a in enumerate(Y_predicted):
                if a[0] == Y_test[idx]:
                    right += 1
                total +=1
            scores[it] = float(right) / float(total)
            print scores[it]
            it += 1   
        scores = np.array(scores)
        return scores.mean()

def delete_rels(sentences_array,r_types,trainset_p):
    print "deleting relations..."
    r_deleted = {}
    #print self.r_types
    for rt in r_types:
        for r in r_types[rt]:
            #print random.random() < trainset_p
            if random.random() < trainset_p:
                #print "entra!"
                for s in sentences_array:
                    if s[0] == r["s"] and r["t"] in s:
                        s.remove(r["t"])
                    if s[0] == r["t"] and r["s"] in s:
                        s.remove(r["s"])
                if not rt in r_deleted:
                    r_deleted[rt] = []
                r_deleted[rt].append(r)
    return sentences_array, r_deleted
