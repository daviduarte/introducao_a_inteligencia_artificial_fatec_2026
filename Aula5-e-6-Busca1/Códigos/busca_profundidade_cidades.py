# DFS iterativo com mínimas modificações

def dfs_caminho(graph, start, goal):
    stack = [[start]]   # pilha de caminhos (antes era queue = deque)
    visited = set()

    while stack:
        path = stack.pop()      # tira do topo (antes era queue.popleft())
        city = path[-1]         # última cidade do caminho

        if city == goal:
            return path         # caminho encontrado

        if city not in visited:
            visited.add(city)

            for neighbor in graph[city]:
                new_path = path + [neighbor]
                stack.append(new_path)   # antes era queue.append()

    return None  # caso não exista caminho


# Grafo realista simples
cidades = {
    "Campinas": ["Jundiaí", "Indaiatuba"],
    "Jundiaí": ["São Paulo"],
    "Indaiatuba": ["São Paulo"],
    "São Paulo": []
}

caminho = dfs_caminho(cidades, "Campinas", "São Paulo")
print("Caminho encontrado:", caminho)
