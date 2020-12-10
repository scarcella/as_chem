import torch 

class graph_inputs(object):
    def __init__(self,input_list):
        self.X = input_list[0]
        self.A = input_list[1]
        self.y = input_list[2]
        

class batch_graph():
    def __init__(self, inputs):
        Xs = [b.X for b in inputs]
        As = [b.A for b in inputs]
        Ys = [b.y for b in inputs]
        
        self.A = torch.block_diag(*As).cuda()
        self.X = torch.cat(Xs).cuda()
        self.Y = Ys
        self.graph_sizes = [len(x) for x in Xs]
        

def graph_collate(batch):
    inputs = batch_graph([graph_inputs(sample) for sample in batch])
    return inputs
