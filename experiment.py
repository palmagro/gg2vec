from node2vec import *
import sys    # sys.setdefaultencoding is cancelled by site.py
reload(sys)    # to re-enable sys.setdefaultencoding()
sys.setdefaultencoding('utf-8')



class experiment:
    def __init__(self,bd,port,user,pss,label,param,trainset_p):
        if param == "ns":
            for i in range(1,5):
                n2v = node2vec(bd,port,user,pss,label,i*1000,200,6,"normal")
                n2v.delete_rels(int(trainset_p))
                n2v.learn("normal")
                n2v.link_prediction_ratio()
