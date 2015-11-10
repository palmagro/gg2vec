from matplotlib import pyplot as plt

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

def all_figure(n2v):
    pal = pallete("nodes") 
    #mds = manifold.TSNE(n_components=2)
    mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')
    X = []
    Y = []
    C = []
    for idx,nt in enumerate(n2v.n_types):
        for n in n2v.n_types[nt]:
            if n in n2v.w2v:
                X.append(n2v.w2v[n])
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
    o = figure(title="All Nodes",plot_height=n2v.ploth,plot_width=n2v.plotw)
    print "d"
    print c
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

