import pickle

# import networkx as netx

with open(
    "data/academic_ex_data/line_416/distributed.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    R = pickle.load(file)
    TD = pickle.load(file)
    b = pickle.load(file)
    f = pickle.load(file)
    V0 = pickle.load(file)
    bounds = pickle.load(file)
    A = pickle.load(file)
    B = pickle.load(file)
    A_cs = pickle.load(file)
