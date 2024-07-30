import typing as T
from math import inf

import torch
from torch.nn.functional import pad
from torch import Tensor

# def construct_graph(scores): #input would be list of list that represents 1 sentence's arc scores
#         """"""
#         graph = {}
#         # Iterating over each vertex in the sentence (including ROOT)
#         for j in range(scores.size(0)): #num rows
#             graph[j] = {}
#             # Iterating over possible heads for vertex j
#             for i in range(scores.size(1)): #num column
#                 if i != j:  # Exclude self-loops
#                     weight = scores[i, j].item()
#                     graph[j][i] = weight
#         print(f"graph: {graph}\n")
#         return graph
def construct_graph(scores): #input would be torch that represents 1 sentence's arc scores
        graph = {}
        for j in range(scores.size(0)): #num rows
            graph[j] = {}
            for i in range(scores.size(1)): #num column
                if i != j:  # Exclude self-loops
                    weight = scores[i, j].item()
                    graph[j][i] = weight
        print(f"graph: {graph}\n")
        return graph
if __name__ == '__main__':
     construct_graph(torch.tensor(
            [[0, 0, 0, 0],
              [12, 0, 6, 5],
              [4, 5, 0, 7],
              [4, 7, 8, 0]]))