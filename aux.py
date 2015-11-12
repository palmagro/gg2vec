import random
import numpy as np
from bokeh.charts.utils import cycle_colors
import seaborn as sns
def dotproduct(a,b):
	return sum([a[i]*b[i] for i in range(len(a))])

from math import acos
import math 
colormapn = ["#EF4136","#FCAF17","#682F79","#1C75BC","#EF2A7B","#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c","#fb9a99","FF6600"]
colormap2 = [
    "#fff9d8",
"#ffe8cd",
"#dbc0ae",
"#cccccc",
"#999999",
"#3252b2"]
colormapa = [
    "#58dc91","#52daca","#f05574","#e1b560","#6c49da","#ff09d8","#BCF1ED", "#999999", "#ff7f00", "#cab2d6", "#6a3d9a",
"#ffe8cd",
"#dbc0ae",
"#cccccc",
"#999999",
"#3252b2"]
colormapa2 = [
    "#58dc95","#52dace","#f05578","#e1b565","#6c49de","#ff09dc","#BCF1ED", "#99999e", "#ff7f05", "#cab2db", "#6a3d9a",
"#ffe8cd",
"#dbc0ae",
"#cccccc",
"#999999",
"#3252b2"]
#e9d9af

colormap = [
    
    "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"
]

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

def pallete(t):
    test = {}
    for i in range(0,100):
        test[str(i)] = np.random.normal(0,1,100)
    if t == "nodes":
        return cycle_colors(test,palette=colormapn)
    else:
        if t == "desv":
            return cycle_colors(test,palette=colormapa2)
        else:
            return cycle_colors(test,palette=colormapa)        

def rotatePoint(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point

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
