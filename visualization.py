from matplotlib import pyplot as plt
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

