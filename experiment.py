import node2vec

sys.setdefaultencoding('utf-8')

class experiment:

    def __init__(self,bd,port,user,pss,label,param,trainset_size):

        if param == "ns":
            for i in range(1,10):
                n2v = node2vec(b,port,user,pss,label,i*1000,200,6,"degree")
                n2v.delete_rels(int(trainset_size*len(sum(len(v) for v in n2v.r_types.itervalues()))))
        self.bd = bd
