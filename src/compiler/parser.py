import numpy as np


class GraphParser:
    '''
    This is a class for parsing input graph G.
    Wrapping various kinds of notation of graph.
    (adjacency matrix, connectivity, etc...)
    '''
    def __init__(self, graph, walk_type='szegedy', optimize=True):
        '''
        args:
            optimize: boolean choosing if you apply problem optimization
                      with matrix transformation.
        '''
        wtype = ['szegedy']
        if walk_type not in wtype:
            raise ValueError('%s is not supported yet' % str(walk_type))
        
        if isinstance(graph, list):
            self.graph = np.array(graph)

    def dim(self):
        # return dimension
        return len(self.graph)

    def _is_directed(self):
        if self.graph.tolist() == self.graph.T.tolist():
            return True
        else:
            return False

    def refence_state(self):
        pass
