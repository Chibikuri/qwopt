import numpy as np
from collections import Counter


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
        elif isinstance(graph, np.ndarray):
            self.graph = graph
        else:
            # FIXME add connection list
            raise ValueError('Invalid format of graph')
        
        if optimize:
            self.graph_opt

    def __len__(self):
        return len(self.graph)

    def n_connections(self):
        return np.sum(self.graph, axis=0)

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
        sorted_connections = np.argsort(np.sort(n_connections))
        argso = np.argsort(n_connections)
        # FIXME looking for more efficient way
        tgraph = np.zeros(self.dim())
        fgraph = np.zeros(self.dim())
        conv_map = []
        for sc, ac in zip(sorted_connections, argso):
            if (ac, sc) not in conv_map:
                conv_map.append((sc, ac))
        for cv in conv_map:
            tgraph[:, cv[0]] = self.graph[:, cv[1]]
            tgraph[:, cv[1]] = self.graph[:, cv[0]]
        for cv in conv_map:
            fgraph[cv[0], :] = tgraph[cv[1], :]
            fgraph[cv[1], :] = tgraph[cv[0], :]
        self.graph = fgraph
        # return for test
        return self.graph
        
    def _matrix_chref_sort(self):
        '''
        function for changing reference state for reducing T operation
        '''
        n_connections = self.n_connections
        print(n_connections)

    def reference_state(self):
        pass


# if __name__ == '__main__':
#     # matrix = np.array([[0, 1, 1, 0],
#     #                    [0, 0, 1, 1],
#     #                    [0, 0, 0, 1],
#     #                    [0, 1, 1, 0]])
#     matrix = np.array([[0, 1, 1, 0, 0, 1],
#                        [0, 0, 1, 1, 1, 1],
#                        [0, 0, 0, 1, 0, 0],
#                        [0, 1, 1, 0, 1, 0],
#                        [0, 1, 1, 0, 1, 0],
#                        [0, 1, 1, 0, 1, 0]])
#     gparser = GraphParser(matrix).matrix_opt()
