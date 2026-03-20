import heapq  # Biblioteca que fornece uma FILA DE PRIORIDADE (pega sempre o menor valor primeiro)

# -------------------------------------------------------------------------
# GRAFO DAS CIDADES
# -------------------------------------------------------------------------
# Cada cidade é uma "chave" do dicionário.
# Para cada cidade, definimos para onde é possível ir e o custo (distância fictícia).
# Exemplo:
# "Campinas": {"Valinhos": 10, "Sumaré": 18}
# Significa:
#   - De Campinas até Valinhos custa 10
#   - De Campinas até Sumaré custa 18
#
graph = {
    "Campinas": {"Valinhos": 10, "Sumaré": 18},
    "Valinhos": {"Vinhedo": 8, "Jundiaí": 24},
    "Sumaré": {"Americana": 12},
    "Americana": {"Jundiaí": 30},
    "Vinhedo": {"Jundiaí": 14},
    "Jundiaí": {"São Paulo": 40},
    "São Paulo": {}  # São Paulo não leva a mais nenhum lugar
}

# -------------------------------------------------------------------------
# HEURÍSTICA
# -------------------------------------------------------------------------
# A heurística é uma "estimativa de distância" até o objetivo (São Paulo).
# Esses valores são fictícios.
# Quanto menor o valor, mais "perto" estamos de São Paulo.
#
heuristica = {
    "Campinas": 90,
    "Valinhos": 70,
    "Vinhedo": 55,
    "Sumaré": 95,
    "Americana": 110,
    "Jundiaí": 50,
    "São Paulo": 0  # O destino sempre tem heurística 0
}

# -------------------------------------------------------------------------
# ALGORITMO A*
# -------------------------------------------------------------------------
# graph -> grafo com as cidades
# start -> cidade inicial
# goal -> destino
# h -> heurística
#
def a_star(graph, start, goal, h):

    # FILA DE PRIORIDADE
    # A fila guarda tuplas do tipo: (f, g, caminho)
    # f -> custo estimado total (g + heurística)
    # g -> custo real acumulado até agora
    # caminho -> lista com as cidades percorridas
    fila = []

    # Inserimos o estado inicial na fila:
    # - f = h[start] (pois g = 0 no começo)
    # - g = 0
    # - caminho = [start]
    heapq.heappush(fila, (h[start], 0, [start]))

    # Conjunto para marcar cidades já visitadas
    visited = set()

    # ---------------------------------------------------------------------
    # LOOP PRINCIPAL DO A*
    # Enquanto houver itens na fila de prioridade...
    # ---------------------------------------------------------------------
    while fila:

        # Remove da fila o estado de MENOR custo f.
        # Isso é automático graças ao heapq.
        f, g, path = heapq.heappop(fila)

        # Última cidade do caminho atual
        city = path[-1]

        # Se chegamos ao objetivo, devolvemos o caminho encontrado!
        if city == goal:
            return path, g  # g é o custo real total

        # Se essa cidade já foi visitada, ignoramos
        if city in visited:
            continue

        # Marcamos a cidade como visitada
        visited.add(city)

        # -----------------------------------------------------------------
        # EXPANSÃO DOS VIZINHOS
        # Para cada cidade vizinha (conectada)
        # -----------------------------------------------------------------
        for neighbor, cost in graph[city].items():

            # Calcula o novo custo real (g_new)
            g_new = g + cost

            # Calcula o novo custo estimado total (f_new = g_new + heurística)
            f_new = g_new + h[neighbor]

            # Cria o novo caminho adicionando o vizinho
            new_path = path + [neighbor]

            # Insere o novo estado na fila
            heapq.heappush(fila, (f_new, g_new, new_path))

    # Se a fila esvaziar e não encontramos o destino
    return None


# -------------------------------------------------------------------------
# EXECUÇÃO DO A*
# -------------------------------------------------------------------------
caminho, custo = a_star(graph, "Campinas", "São Paulo", heuristica)

print("Caminho encontrado:", caminho)
print("Custo real total:", custo)
