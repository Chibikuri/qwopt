import numpy as np
from collections import Counter


class GraphParser:
    '''
    This is a class for parsing input graph G.
    Wrapping various kinds of notation of graph.
    (adjacency matrix, connectivity, etc...)
    '''
    def __init__(self, graph, prob_tran, walk_type='szegedy', optimize=True):
        '''
        args:
            prob_tran: probability transition matrix
            optimize: boolean choosing if you apply problem optimization
                      with matrix transformation.

        '''
        wtype = ['szegedy']
        if walk_type not in wtype:
            raise ValueError('%s is not supported yet' % str(walk_type))

        if isinstance(graph, list):
            self.graph = np.array(graph)
        elif isinstance(graph, np.ndarray):
            self.graph = graph
        else:
            # FIXME add connection list
            raise ValueError('Invalid format of graph')

        if isinstance(prob_tran, list):
            self.ptrans = np.array(prob_tran)
        elif isinstance(prob_tran, np.ndarray):
            self.ptrans = prob_tran
        else:
            # FIXME add connection list
            raise ValueError('Invalid format of transition matrix')

        if optimize:
            self.graph_opt()
            self.opt_flag = True
        
    def __len__(self):
        return len(self.graph)

    def n_connections(self):
        return np.sum(self.graph, axis=0)
    
    def graph(self):
        return self.graph

    def n_partitions(self):
        col_sum = np.sum(self.graph, axis=0)
        return len(set(col_sum))

    def dim(self):
        # return dimension
        return self.graph.shape

    def _is_directed(self):
        if self.graph.tolist() == self.graph.T.tolist():
            return True
        else:
            return False

    def graph_opt(self):
        n_connections = self.n_connections()
        argso = np.argsort(n_connections)
        # FIXME looking for more efficient way
        tgraph = np.zeros(self.dim())
        fgraph = np.zeros(self.dim())
        # FIXME 
        pgraph = np.zeros(self.dim())
        qgraph = np.zeros(self.dim())

        conv_map = []
        for ircv, rcv in enumerate(argso):
            tgraph[:, ircv] = self.graph[:, rcv]
            pgraph[:, ircv] = self.ptrans[:, rcv]
        for iccv, ccv in enumerate(argso):
            fgraph[iccv, :] = tgraph[ccv, :]
            qgraph[iccv, :] = pgraph[ccv, :]
        self.graph = fgraph
        self.ptrans = qgraph
        # return for test
        return self.graph, self.ptrans

    def _matrix_chref_sort(self):
        '''
        function for changing reference state for reducing T operation
        '''
        n_connections = self.n_connections
        # TODO implement sorter inside of connections

    def reference_state(self):
        if self.opt_flag:
            n_connections = self.n_connections()
            c_types = set(map(int, n_connections))
            ref_index = [np.where(n_connections == c)[0][0] for c in c_types]
            ref_state = [self.ptrans[:, i] for i in ref_index]
            return ref_state, ref_index
        else:
            # FIXME
            raise Exception('No optimization is not permited yet')


def prob_transition(graph):
    pmatrix = np.zeros(graph.shape)
    indegrees = np.sum(graph, axis=0)
    for ix, indeg in enumerate(indegrees):
        if indeg == 0:
            pmatrix[:, ix] = graph[:, ix]
        else:
            pmatrix[:, ix] = graph[:, ix]/indeg
    return pmatrix


# if __name__ == '__main__':
#     graph = np.array([[0, 1, 0, 0, 1, 0],
#                       [0, 0, 0, 1, 1, 0],
#                       [0, 0, 0, 1, 1, 1],
#                       [0, 1, 1, 0, 0, 0],
#                       [0, 1, 0, 0, 0, 1],
#                       [0, 1, 0, 0, 1, 0]])
#     pb = prob_transition(graph)
#     parser = GraphParser(graph, pb)
    # print(parser.graph)
    # print(parser.ptrans)
