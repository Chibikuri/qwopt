import QuantumWalkOptimizer


class GraphParserTest:

    def __init__(self):
        pass

    def dim_test(self):
        test_mat = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        gparser = GraphParser(test_mat)
        gdim = gparser.dim()
        print(gdim)
        assert(gdim==(4, 4))


if __name__ == '__main__':
    gparsetest = GraphParserTest()
    gparsetest.dim_test()