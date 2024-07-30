#!/usr/bin/env python3
# Student name: Xueqing Zhai
# Student number: 1006962413
# UTORid: zhaixueq
# Note: a part of single_root_mst in this file is not my work. I added that part only to make sure other parts could run. 

import typing as T
from math import inf

import torch
from torch.nn.functional import pad
from torch import Tensor


def is_projective(heads: T.Iterable[int]) -> bool:
    """
    Determines whether the dependency tree for a sentence is projective.

    Args:
        heads: The indices of the heads of the words in sentence. Since ROOT
          has no head, it is not expected to be part of the input, but the
          index values in heads are such that ROOT is assumed in the
          starting (zeroth) position. See the examples below.

    Returns:
        True if and only if the tree represented by the input is
          projective.

    Examples:
        The projective tree from the assignment handout:
        >>> is_projective([2, 5, 4, 2, 0, 7, 5, 7])
        True

        The non-projective tree from the assignment handout:
        >>> is_projective([2, 0, 2, 2, 6, 3, 6])
        False
    """
    projective = True
    # *** BEGIN YOUR CODE *** #
    for i in range(len(heads)):
        left = min(i+1, heads[i])
        right = max(i+1, heads[i])
        for j in range(len(heads)):
            # print(f"left, right: {left}, {right}\n")
            # print(f"min, max: {min(j+1, heads[j])}, {max(j+1, heads[j])}\n")
            if (min(j+1, heads[j]) < right and min(j+1, heads[j]) > left) and (max(j+1, heads[j]) > right):
                # print("1enter here\n")
                return False
            if (min(j+1, heads[j] < left)) and (max(j+1, heads[j]) > left and max(j+1, heads[j]) < right):
                # print("2enter here\n")
                return False
    # *** END YOUR CODE *** #
    return projective


def is_single_root_tree(heads: Tensor, lengths: Tensor) -> Tensor:
    """
    Determines whether the selected arcs for a sentence constitute a tree with
    a single root word.

    Remember that index 0 indicates the ROOT node. A tree with "a single root
    word" has exactly one outgoing edge from ROOT.

    If you like, you may add helper functions to this file for this function.

    This file already imports the function `pad` for you. You may find that
    function handy. Here's the documentation of the function:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

    Args:
        heads (Tensor): a Tensor of dimensions (batch_sz, sent_len) and dtype
            int where the entry at index (b, i) indicates the index of the
            predicted head for vertex i for input b in the batch

        lengths (Tensor): a Tensor of dimensions (batch_sz,) and dtype int
            where each element indicates the number of words (this doesn't
            include ROOT) in the corresponding sentence.

    Returns:
        A Tensor of dtype bool and dimensions (batch_sz,) where the value
        for each element is True if and only if the corresponding arcs
        constitute a single-root-word tree as defined above

    Examples:
        Valid trees from the assignment handout:
        >>> is_single_root_tree(torch.tensor([[2, 5, 4, 2, 0, 7, 5, 7],\
                                              [2, 0, 2, 2, 6, 3, 6, 0]]),\
                                torch.tensor([8, 7]))
        tensor([True, True])

        Invalid trees (the first has a cycle; the second has multiple roots):
        >>> is_single_root_tree(torch.tensor([[2, 5, 4, 2, 0, 8, 6, 7],\
                                              [2, 0, 2, 2, 6, 3, 6, 0]]),\
                                torch.tensor([8, 8]))
        tensor([False, False])
    """
    # *** BEGIN YOUR CODE *** #
    def construct_graph(row):
        graph = {}
        for idx, head in enumerate(row):
            if head.item() not in graph:
                graph[head.item()] = []
            graph[head.item()].append(idx + 1)
        vertices = set(range(len(row) + 1))
        # print(f"graph: {graph}\n")
        return graph, vertices
    
    def is_tree(graph, vertices):
        visited = {v: False for v in vertices}
        parent = {v: None for v in vertices}
        # from 263
        def dfs(v, visited, parent):
            visited[v] = True
            for u in graph.get(v, []):
                if not visited[u]:
                    parent[u] = v
                    if dfs(u, visited, parent):
                        return True
                elif parent[v] != u:
                    return True
            return False

        contains_cycle = dfs(0, visited, parent)

        all_visited = all(visited[v] for v in vertices)
        # print(f"cycle: {contains_cycle}\n")
        # print(f"visited: {all_visited}\n")
        num_edge = sum(len(graph[key]) for key in graph.keys())
        return (num_edge == len(vertices) - 1) and not contains_cycle and all_visited

    tree_single_root = torch.ones_like(heads[:, 0], dtype=torch.bool)
    for i, sentence in enumerate(heads):
        tree = is_tree(*construct_graph(sentence[:lengths[i]]))
        tree_single_root[i] = tree
        count_0 = (sentence[:lengths[i]] == 0).sum().item()
        if count_0 != 1:
            tree_single_root[i] = False

    return tree_single_root



def single_root_mst(arc_scores: Tensor, lengths: Tensor) -> Tensor:
    """
    Finds the maximum spanning tree (more technically, arborescence) for the
    given sentences such that each tree has a single root word.

    Remember that index 0 indicates the ROOT node. A tree with "a single root
    word" has exactly one outgoing edge from ROOT.

    If you like, you may add helper functions to this file for this function.

    This file already imports the function `pad` for you. You may find that
    function handy. Here's the documentation of the function:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

    Args:
        arc_scores (Tensor): a Tensor of dimensions (batch_sz, x, y) and dtype
            float where x=y and the entry at index (b, i, j) indicates the
            score for a candidate arc from vertex j to vertex i.

        lengths (Tensor): a Tensor of dimensions (batch_sz,) and dtype int
            where each element indicates the number of words (this doesn't
            include ROOT) in the corresponding sentence.

    Returns:
        A Tensor of dtype int and dimensions (batch_sz, x) where the value at
        index (b, i) indicates the head for vertex i according to the
        maximum spanning tree for the input graph.

    Examples:
        >>> single_root_mst(torch.tensor(\
            [[[0, 0, 0, 0],\
              [12, 0, 6, 5],\
              [4, 5, 0, 7],\
              [4, 7, 8, 0]],\
             [[0, 0, 0, 0],\
              [1.5, 0, 4, 0],\
              [2, 0.1, 0, 0],\
              [0, 0, 0, 0]],\
             [[0, 0, 0, 0],\
              [4, 0, 3, 1],\
              [6, 2, 0, 1],\
              [1, 1, 8, 0]]]),\
            torch.tensor([3, 2, 3]))
        tensor([[0, 0, 3, 1],
                [0, 2, 0, 0],
                [0, 2, 0, 2]])
    """
    # *** BEGIN YOUR CODE *** #
    # best_arcs = arc_scores.argmax(-1)
    def construct_graph(scores): #input would be torch that represents 1 sentence's arc scores
        graph = {}
        for j in range(scores.size(0)): #num rows
            graph[j] = {}
            for i in range(scores.size(1)): #num column
                if i != j and i != 0:  # Exclude self-loops and 0 head
                    weight = scores[i, j].item()
                    graph[j][i] = weight
        #print(f"graph: {graph}\n")
        return graph
    
    ### IMPORTANT: I looked at the Chu-liu Edmonds algorithm from: 
    # https://wendy-xiao.github.io/posts/2020-07-10-chuliuemdond_algorithm/
    # because I'm out of time and just want it to be able to run, in order to test other parts. 
    # My implementation below is almost identical as the code from the link. Thus this part is not my work. 
    def reverse_graph(G):
        g={}
        for src in G.keys():
            for dst in G[src].keys():
                if dst not in g.keys():
                    g[dst]={}
                g[dst][src]=G[src][dst]
        return g

    def build_max(rg,root):
        max_g = {}
        for dst in rg.keys():
            if dst==root:
                continue
            max_ind=-100
            max_value = -100
            for src in rg[dst].keys():
                if rg[dst][src]>=max_value:
                    max_ind = src
                    max_value = rg[dst][src]
            max_g[dst]={max_ind:max_value}
        return max_g

    def find_circle(max_g):            
        for start in max_g.keys():
            visited=[]
            stack = [start]
            while stack:
                n = stack.pop()
                if n in visited:
                    C = []
                    while n not in C:
                        C.append(n)
                        n = list(max_g[n].keys())[0]
                    return C
                visited.append(n)
                if n in max_g.keys():
                    stack.extend(list(max_g[n].keys()))
        return None
            
    def chu_liu_edmond(G,root):
        rg = reverse_graph(G)
        rg[root]={}
        max_g = build_max(rg,root)
        
        C = find_circle(max_g)
        if not C:
            return reverse_graph(max_g)
        all_nodes = G.keys()
        vc = max(all_nodes)+1
        
        V_prime = list(set(all_nodes)-set(C))+[vc]
        G_prime = {}
        vc_in_idx={}
        vc_out_idx={}
        for u in all_nodes:
            for v in G[u].keys():

                if (u not in C) and (v in C):
                    if u not in G_prime.keys():
                        G_prime[u]={}
                    w = G[u][v]-list(max_g[v].values())[0]
                    if (vc not in  G_prime[u]) or (vc in  G_prime[u] and w > G_prime[u][vc]):
                        G_prime[u][vc] = w
                        vc_in_idx[u] = v

                elif (u in C) and (v not in C):
                    if vc not in G_prime.keys():
                        G_prime[vc]={}
                    w = G[u][v]
                    if (v not in  G_prime[vc]) or (v in  G_prime[vc] and w > G_prime[vc][v]):
                        G_prime[vc][v] = w
                        vc_out_idx[v] = u
                elif (u not in C) and (v not in C):
                    if u not in G_prime.keys():
                        G_prime[u]={}
                    G_prime[u][v] = G[u][v]

        A = chu_liu_edmond(G_prime,root)
        all_nodes_A = list(A.keys())
        for src in all_nodes_A:
            if src==vc:
                for node_in in list(A[src].keys()):
                    orig_out = vc_out_idx[node_in]
                    if orig_out not in A.keys():
                        A[orig_out] = {}
                    A[orig_out][node_in]=G[orig_out][node_in]
            else:
                # edited this part from source so dictionary won't be modified in loop
                for dst in list(A[src].keys()):
                    if dst==vc:
                        orig_in = vc_in_idx[src]
                        A[src][orig_in] = G[src][orig_in]
                        del A[src][dst]
        # modified this part so no eror if vc not in A
        A.pop(vc, None)
        
        for node in C:
            if node != orig_in:
                src = list(max_g[node].keys())[0]
                if src not in A.keys():
                    A[src] = {}
                A[src][node] = max_g[node][src]
        return A 
    ### Chu-liu Edmonds part end. The rest are my code

    batch_size = arc_scores.size(0)
    best_arcs = []

    for b in range(batch_size):
        heads = [0] * (lengths[b].item() + 1)
        sentence_graph = construct_graph(arc_scores[b, :lengths[b]+1, :lengths[b]+1])
        # find mst
        mst = chu_liu_edmond(sentence_graph, 0)
        single_root = len(mst.get(0, {}))
        if single_root != 1:
            best_score = float('-inf')
            best_mst = mst

            for potential_root in range(1, lengths[b].item() + 1): 
                subgraph = {k: v.copy() for k, v in sentence_graph.items() if k != 0}
                potential_mst = chu_liu_edmond(subgraph, potential_root)
                potential_mst[0] = {potential_root: sentence_graph[0][potential_root]}
                score = sum([value for dst_map in potential_mst.values() for value in dst_map.values()])

                if score > best_score:
                    best_score = score
                    best_mst = potential_mst
            mst = best_mst

        for src, dst_map in mst.items():
            for dst in dst_map.keys():
                heads[dst] = src
        # print(f"heads: {heads}\n")
        best_arcs.append(heads)

    max_length = max(map(len, best_arcs))
    best_arcs = [arc + [0] * (max_length - len(arc)) for arc in best_arcs]
    # print(f"best_arcs: {best_arcs}\n")

    result = torch.tensor(best_arcs)
    # print(f"return: {result}\n")
    return result
    # *** END YOUR CODE *** #


if __name__ == '__main__':
    import doctest
    doctest.testmod()
