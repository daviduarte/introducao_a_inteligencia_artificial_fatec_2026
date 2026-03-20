# BFSfrom collections import deque
"""
BFS usa uma FILA → FIFO
from collections import deque
queue = deque()
queue.append(item)   # entra no fim
queue.popleft()      # sai do início
"""


from collections import deque

def bfs_caminho(graph, start, goal):
    queue = deque([[start]])   # fila de caminhos
    visited = set()

    while queue:
        path = queue.popleft()    # pega o caminho atual
        city = path[-1]           # última cidade do caminho

        if city == goal:
            return path           # caminho encontrado

        if city not in visited:
            visited.add(city)

            for neighbor in graph[city]:
                new_path = path + [neighbor]  # cria extensão do caminho
                queue.append(new_path)

    return None  # caso não exista caminho


# Grafo realista simples
cidades = {
    "Campinas": ["Jundiaí", "Indaiatuba"],
    "Jundiaí": ["São Paulo"],
    "Indaiatuba": ["São Paulo"],
    "São Paulo": []
}

caminho = bfs_caminho(cidades, "Campinas", "São Paulo")
print("Caminho encontrado:", caminho)
