import numpy as np
from math import acos
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

def generate_sample(mode,sentences_array,degree,w_size):
    if mode == "degree":
        s = sentences_array[weighted_choice(degree)]
    else:
        s = np.random.choice(sentences_array)
    s = eval(str(s))               
    a = s[0] 
    b = sample_wr(s[1:],w_size)
    b.insert(0,a)
    return b

import random

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
