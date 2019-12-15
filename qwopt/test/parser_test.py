import sys
sys.path.append('../../')
import unittest
import numpy as np
from qwopt.compiler.parser import GraphParser


class GraphParserTest(unittest.TestCase):

    def test_dim(self):
        test_mat = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        gparser = GraphParser(test_mat)
        gdim = gparser.dim()
        self.assertEqual(gdim, (4, 4), 'Unexpected')

    def test_len(self):
        test_mat = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        gparser = GraphParser(test_mat)
        glen = len(gparser)
        self.assertEqual(glen, 4, 'Unexpected')

    def test_n_partition(self):
        test_mat = [[0, 0, 1, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 1],
                    [1, 1, 1, 1]] 
        gparser = GraphParser(test_mat)
        n_partition = gparser.n_partitions()
        self.assertEqual(n_partition, 3, 'number of partitions')

    def test_n_connections(self):
        test_mat = [[0, 0, 1, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 1],
                    [1, 1, 1, 1]] 
        gparser = GraphParser(test_mat)
        gpartition = gparser.n_connections()
        bolmx = gpartition == [3, 2, 3, 4]
        self.assertEqual(all(bolmx), True, 'number of connections')

    def test_matrix_opt(self):
        test_mat = [[0, 1, 1, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 1, 1, 0]]
        gparser = GraphParser(test_mat)
        gmatrix = gparser.graph_opt()
        cmatrix = np.array([[0, 1, 1, 1],
                            [0, 0, 0, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0]])
        bolmx = gmatrix == cmatrix
        judger = [all(i) for i in bolmx]
        self.assertEqual(all(judger), True, 'matrix optimization')


if __name__ == '__main__':
    unittest.main()
