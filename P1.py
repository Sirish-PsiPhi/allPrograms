from heuristicsearch.a_star_search import AStar

# For Example 1
adjacency_list = {
    'A': [('B', 1), ('C', 3), ('D', 7)],
    'B': [('D', 5)],
    'C': [('D', 12)]
}

heuristics = {'A':1, 'B':1, 'C':1, 'D':1}

graph = AStar(adjacency_list, heuristics)
graph.apply_a_star(start='A',stop='D')

#For Example 2
adjacency_list = {
    'S': [('A', 1), ('G', 10)],
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 5)],
    'C': [('D', 3), ('G', 4)],
    'D': [('C', 3), ('G', 2)]
}

heuristics = {'S':5, 'A':3, 'B':4, 'C':2, 'D':6, 'G':0}

graph = AStar(adjacency_list, heuristics)
graph.apply_a_star(start='S',stop='G')
