from node2vec import *
import sys    # sys.setdefaultencoding is cancelled by site.py
reload(sys)    # to re-enable sys.setdefaultencoding()
sys.setdefaultencoding('utf-8')
from experiment import *


class composite_experiment:
    def __init__(self,ds,param,trainset_p,iteraciones):
        self.ds = ds
        self.param = param
        self.p = figure(plot_width=450, plot_height=240)    
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
            self.p.legend.background_fill_alpha = 0.5

    def ltype_prediction(self,a,b,jump,dev):
        pal = pallete("db")
        for idx,d in enumerate(self.ds):
            e = experiment(d[0],7474,user,pswd,d[1],"normal",self.param,self.trainset_p,self.iteraciones)
            x,y,xd,yd = e.ltype_prediction(a,b,jump)
            self.p.line(x, y, color=pal[idx],legend=d[2],line_width=2.0)
            if(dev):
                self.p.line(xd, yd, color=pal[idx],legend=d[2] + "dev",line_width=2.0,line_dash='dotted')
            self.p.legend.background_fill_alpha = 0.5        
