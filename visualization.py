import csv
from matplotlib import pyplot as plt
import random
import numpy as np
from bokeh.charts.utils import cycle_colors
from sklearn import manifold
from math import acos
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show, ColumnDataSource, vplot
from bokeh.models import(
    GMapPlot, Range1d, ColumnDataSource, LinearAxis,
    PanTool, WheelZoomTool, BoxZoomTool, ResetTool, ResizeTool, BoxSelectTool, HoverTool)#printfTickFormatter)
from bokeh.charts import Line
import math 
from aux import *
from node2vec import *
import colorsys

colormapn = ["#1C75BC","#FCAF17","#EF4136","#682F79","#a6cee3", "#444444", "#1f78b4", "#b2df8a", "#33a02c","#fb9a99","FF6600"]
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
"#3252b2","#FA5CE5","#DEFACE"]
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
            if t == "traversals":
                return cycle_colors(test,palette=colormap)  
            else:
                return cycle_colors(test,palette=colormapa)  

def links_figure(n2v):
    pal = pallete("links") 
    #Links Plotting
    q = figure(title="Link Labels")
    for idx,r in enumerate(n2v.r_types):
        x = np.linspace(math.pi+math.pi/2,3*math.pi+math.pi/2,2000)
        cdf = np.sin(x)*len(n2v.r_types[r])/2
        q.line(x*500000*n2v.r_types_d[idx] + idx*1000000, cdf+(len(n2v.r_types[r])/2), alpha=0.9,color=pal[idx],line_width=1)
        q.patch(x*500000*n2v.r_types_d[idx] + idx*1000000, cdf+(len(n2v.r_types[r])/2), alpha=0.7, legend=r,color=pal[idx])
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
    for i, n1 in enumerate(n2v.r_types):
        names.append(n1)
        for j, n2 in enumerate(n2v.r_types):
            xname.append(n1)
            yname.append(n2)
            alpha.append(n2v.angle_matrix[n1][n2]/math.pi)
            angle.append(np.degrees(n2v.angle_matrix[n1][n2]))

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

def all_figure(n2v,tp,ntypes,legend):
    pal = pallete("nodes") 
    X = []
    Y = []
    C = []
    for idx,nt in enumerate(ntypes):
        for n in n2v.n_types[nt]:
            if random.random() < tp[idx]:    
                if n in n2v.w2v:
                    X.append(n2v.w2v[n])
                    Y.append(n)
                    C.append(idx)
    mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
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
    o = figure(plot_height=n2v.ploth,plot_width=n2v.plotw)
    for idx,nt in enumerate(ntypes):
        o.line([], [], color=pal[idx],legend=nt,line_width=1.5)
    if legend:
        o.text('x', 'y', label, source=source, )
    o.circle('x', 'y', size=10, source=source,color=c,alpha=0.5 )
    return o

def all_links_figure(n2v,tp,ltypes,legend):
    showed = []
    pal = pallete("links") 
    #mds = manifold.TSNE(n_components=2)
    mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
    X = []
    C = []
    for idx,nt in enumerate(ltypes):
        for a in n2v.r_types[nt]:
            if random.random() < tp[idx]:    
                X.append(a["v"])
                C.append(idx)

    result = mds.fit_transform(np.asfarray(X,dtype='float'))
    x = []
    y = []
    c = []
    label = []
    for idx,v in enumerate(result):
        x.append(result[idx][0])
        y.append(result[idx][1])
        c.append(pal[C[idx]])

    source = ColumnDataSource(data=dict(x=x,y=y, label=label))
    #Nodes Plotting
    o = figure(title="All Links",plot_height=n2v.ploth,plot_width=n2v.plotw)
    for idx,rt in enumerate(n2v.r_types):
        if rt in ltypes:   
            o.line([], [], color=pal[idx],legend=rt,line_width=1.5)
    o.text('x', 'y', label, source=source, )
    o.circle('x', 'y', size=10, source=source,color=c )
    return o

def some_figure(n2v,c,legend):
    mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
    X = []
    Y = []
    C = []
    con = neo4j.CypherQuery(n2v.graph_db, c).execute()
    for row in con:
        if row[2] in n2v.w2v:
            print row[0]
            X.append(n2v.w2v[row[2]])
            Y.append(row[2])
            C.append(ord(row[1][0]) % 10)
        


    pal = pallete("nodes") 
    #mds = manifold.TSNE(n_components=2)
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
    o = figure(title="All Nodes",plot_height=n2v.ploth,plot_width=n2v.plotw)
    if legend:
        o.text('x', 'y', label, source=source, )
    o.circle('x', 'y', size=10, source=source,color=c )
    return o

def nodes_figure(n2v):
    pal = pallete("nodes")
    palr = pallete("links")
    #mds = manifold.TSNE(n_components=2)
    mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
    X = []
    Y = []
    for v in n2v.m_points:
        X.append(n2v.m_points[v])
        Y.append(v)
    #print X
    result = mds.fit_transform(np.asfarray(X,dtype='float'))
    for idx,r in enumerate(Y):
        n2v.m_points[r] = result[idx] 
    x = []
    y = []
    d = []
    d1 = []
    d2 = []
    c=[]
    label = []
    nnodes = []
    for idx,v in enumerate(n2v.m_points):
        label.append(v)
        x.append(n2v.m_points[v][0])
        y.append(n2v.m_points[v][1])
        d.append(n2v.n_types_d[v])
        d1.append(n2v.n_types_d[v]*2*1.33333)
        d2.append(n2v.n_types_d[v]*2)
        nnodes.append(len(n2v.n_types[v]))
        c.append(pal[idx])
    source = ColumnDataSource(data=dict(x=x,y=y, label=label,d=d,nnodes=nnodes))
    #Nodes Plotting
    o = figure(title="Node Labels",plot_width=n2v.plotw,plot_height=n2v.ploth,tools="pan,wheel_zoom,box_zoom,reset,resize,hover")
    print "d"
    print d
    print d1
    print d2
    o.oval('x', 'y', width=d1,height=d2, source=source,alpha=0.7**len(n2v.m_points),height_units="data",width_units="data",color=c  )

    for idx,r in enumerate(n2v.r_types1):
        a1 = rotatePoint(n2v.m_points[n2v.r_types1[r]["s"]],n2v.m_points[n2v.r_types1[r]["t"]],np.degrees(n2v.r_types_d[idx])/2) 
        a2 = rotatePoint(n2v.m_points[n2v.r_types1[r]["s"]],n2v.m_points[n2v.r_types1[r]["t"]],-np.degrees(n2v.r_types_d[idx])/2) 
        o.segment(x0=n2v.m_points[n2v.r_types1[r]["s"]][0],x1=n2v.m_points[n2v.r_types1[r]["t"]][0],y0=n2v.m_points[n2v.r_types1[r]["s"]][1],y1=n2v.m_points[n2v.r_types1[r]["t"]][1],color=palr[idx],line_width=2.5)
        o.segment(x0=n2v.m_points[n2v.r_types1[r]["t"]][0],x1=a1[0],y0=n2v.m_points[n2v.r_types1[r]["t"]][1],y1=a1[1],color=palr[idx],line_width=2.5)
        o.segment(x0=n2v.m_points[n2v.r_types1[r]["t"]][0],x1=a2[0],y0=n2v.m_points[n2v.r_types1[r]["t"]][1],y1=a2[1],color=palr[idx],line_width=2.5)
        #o.segment(x0=a1[0],x1=a2[0],y0=a1[1],y1=a2[1])
        x = [n2v.m_points[n2v.r_types1[r]["s"]][0],a1[0],n2v.m_points[n2v.r_types1[r]["t"]][0],a2[0],n2v.m_points[n2v.r_types1[r]["s"]][0]]
        y = [n2v.m_points[n2v.r_types1[r]["s"]][1],a1[1],n2v.m_points[n2v.r_types1[r]["t"]][1],a2[1],n2v.m_points[n2v.r_types1[r]["s"]][1]]
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

def vis(model,t,x,y,label):
    if t =="bekoh":
        # output to static HTML file
        output_file("lines.html", title="line plot example")
        source = ColumnDataSource(data=dict(x=x,y=y,label=label))
        hover = HoverTool(
                tooltips=[
                    ("(x,y)", "($x, $y)"),
                    ("label", "@label"),
                ]
            )
        p = figure(plot_width=600, plot_height=800, tools=[hover,BoxZoomTool(),PanTool(), WheelZoomTool(), BoxZoomTool(),ResetTool(), ResizeTool(), BoxSelectTool()],title="Mouse over the dots")
        p.circle('x', 'y', size=5, source=source)
        show(p)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for idx,v in enumerate(model.vocab):
            ax.annotate(v, xy=((x[idx],(y[idx]))))
        plt.scatter(x,y)
        plt.grid()
        plt.show()


def show2D(model,t):
    #mds = manifold.TSNE(n_components=2)
    mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
    X = []
    for v in model.vocab:
        X.append(model[v])
    if ndim > 2:
        result = mds.fit_transform(np.array(X))
    else:
        result = np.array(X)
    x = []
    y = []
    label = []
    for idx,v in enumerate(model.vocab):
        label.append(v)
        x.append(result[idx][0])
        y.append(result[idx][1])

    vis(model,t,x,y,label)

def visual_matrix(matriz,color):
    print matriz
    names = []
    xname = []
    yname = []
    color = []
    alpha = []
    confusion = []
    for idx,n in enumerate(matriz):
        if idx != 0:
            names.append(matriz[idx][0])
    for idx1,n1 in enumerate(matriz):
        for idx2,n2 in enumerate(matriz):
            if idx1 != 0 and idx2 != 0:
                xname.append(matriz[idx1][0])
                yname.append(matriz[0][idx2])
                if not color:
                    alpha.append(matriz[idx1][idx2]/100)
                    confusion.append(matriz[idx1][idx2])
                    color.append('black')
                else:
                    color.append('%02x%02x%02x' % colorsys.hls_to_rgb(matriz[idx1][idx2]/100, 0.5, 0.5))
                    print '%02x%02x%02x' % colorsys.hls_to_rgb(matriz[idx1][idx2]/100, 0.5, 0.5)
    print xname
    print yname
    print alpha
    source = ColumnDataSource(
data=dict(
            xname=xname,
            yname=yname,
            colors=color,
            alphas=alpha,
            angles=confusion
        )
    )
    p = figure(x_axis_location="above", tools="resize,hover,save",
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

def latex_matrix(matriz):
    #Poniendo negritas
    for i in range(1,len(matriz)):
        matriz[i][0] =  "\meg{ "+str(matriz[i][0])+"}"
        matriz[0][i] =  "\meg{ "+matriz[0][i]+"}"
    for i in range(1,len(matriz)):
        matriz[i][i] =  "\meg{ "+matriz[i][i]+"}"
    matriz[0][0] = ""
    return matriz

