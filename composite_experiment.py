from node2vec import *
import sys    # sys.setdefaultencoding is cancelled by site.py
reload(sys)    # to re-enable sys.setdefaultencoding()
sys.setdefaultencoding('utf-8')
from experiment import *


class composite_experiment:
    def __init__(self,ds,param,trainset_p,iteraciones):
        self.ds = ds
        self.param = param
        self.p = figure(plot_width=500, plot_height=250)    
        self.trainset_p = trainset_p
        self.iteraciones = iteraciones

    def ntype_prediction(self,a,b,jump,dev):
        pal = pallete("db")
        for idx,d in enumerate(self.ds):
            e = experiment(d[0],7474,user,pswd,d[1],"normal",self.param,self.trainset_p,self.iteraciones)
            x,y,xd,yd = e.ntype_prediction(a,b,jump)
            self.p.line(x, y, color=pal[idx],legend=d[2],line_width=2.0)
            if(dev):
                self.p.line(xd, yd, color=pal[idx],legend=d[2] + "dev",line_width=2.0,line_dash='dotted')
            self.p.legend.background_fill_alpha = 0.7
            self.p.xaxis.axis_label = xaxis_legend[self.param]
            self.p.yaxis.axis_label = 'Accuracy'

    def ltype_prediction(self,a,b,jump,dev):
        pal = pallete("db")
        for idx,d in enumerate(self.ds):
            e = experiment(d[0],7474,user,pswd,d[1],"normal",self.param,self.trainset_p,self.iteraciones)
            x,y,xd,yd = e.ltype_prediction(a,b,jump)
            self.p.line(x, y, color=pal[idx],legend=d[2],line_width=2.0)
            if(dev):
                self.p.line(xd, yd, color=pal[idx],legend=d[2] + "dev",line_width=2.0,line_dash='dotted')
            self.p.legend.background_fill_alpha = 0.5    
            self.p.xaxis.axis_label = xaxis_legend[self.param]
            self.p.yaxis.axis_label = 'Accuracy'


    def link_prediction(self,traversals,a,b,jump,dev,metrica,filtrado):
        pal = pallete("db")
        for idx,d in enumerate(self.ds):
            e = experiment(d[0],7474,user,pswd,d[1],"normal",self.param,self.trainset_p,self.iteraciones)
            x,y,xd,yd = e.link_prediction(traversals,a,b,jump,metrica,filtrado)
            self.p.line(x, y, color=pal[idx],legend=d[2],line_width=2.0)
            if(dev):
                self.p.line(xd, yd, color=pal[idx],legend=d[2] + "dev",line_width=2.0,line_dash='dotted')
            self.p.legend.background_fill_alpha = 0.5        
            self.p.xaxis.axis_label = xaxis_legend[self.param]
            self.p.yaxis.axis_label = 'MRR'

    def traversal_prediction(self,traversals,a,b,jump,dev,metrica,filtrado):
        pal = pallete("traversals")
        for idx,t in enumerate(traversals):
            e = experiment(self.ds[0],7474,user,pswd,self.ds[1],"normal",self.param,self.trainset_p,self.iteraciones)
            x,y,xd,yd = e.traversal_prediction(t,a,b,jump,metrica,filtrado)
            self.p.line(x, y, color=pal[idx],legend="T"+str(idx+1),line_width=2.0)
            if(dev):
                self.p.line(xd, yd, color=pal[idx],legend=self.ds[2] + "dev",line_width=2.0,line_dash='dotted')
            self.p.legend.background_fill_alpha = 0.5       
            self.p.xaxis.axis_label = xaxis_legend[self.param]
            self.p.yaxis.axis_label = 'MRR'
