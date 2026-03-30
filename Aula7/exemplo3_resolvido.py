from ultralytics import YOLO
from shapely.geometry import box as shapely_box

# ============================================
# 1. FUNÇÃO PARA CRIAR UMA GEOMETRIA SHAPELY
# ============================================

def criar_geometria_caixa(box_xyxy):
    """
    Converte uma bounding box no formato [x1, y1, x2, y2]
    para um retângulo do Shapely.

    Parâmetros:
    - box_xyxy: lista ou tupla no formato [x1, y1, x2, y2]

    Retorno:
    - Um objeto geométrico do Shapely representando a caixa
    """

    x1, y1, x2, y2 = box_xyxy
    return shapely_box(x1, y1, x2, y2)


# ============================================
# 2. FUNÇÃO PARA CALCULAR SOBREPOSIÇÃO
# ============================================

def calcular_sobreposicao(boxA, boxB):
    """
    Calcula a sobreposição entre duas bounding boxes usando Shapely.

    Cada caixa deve estar no formato:
    [x1, y1, x2, y2]

    Retorna:
    - area_intersecao
    - area_boxA
    - area_boxB
    """

    # Cria os retângulos geométricos
    geomA = criar_geometria_caixa(boxA)
    geomB = criar_geometria_caixa(boxB)

    # Calcula a interseção entre os dois retângulos
    intersecao = geomA.intersection(geomB)

    # Áreas
    area_intersecao = intersecao.area
    area_boxA = geomA.area
    area_boxB = geomB.area

    return area_intersecao, area_boxA, area_boxB


# ============================================
# 3. FUNÇÃO PARA VERIFICAR COLISÃO
# ============================================

def ha_colisao(boxA, boxB, limiar=0.10):
    """
    Verifica se duas caixas colidem com base em um limiar de sobreposição.

    Critério usado:
    - calcula a área de interseção
    - divide pela área da menor caixa
    - se o valor for >= limiar, consideramos colisão

    Exemplo:
    - limiar=0.10  -> 10% de sobreposição da menor caixa
    """

    area_intersecao, area_boxA, area_boxB = calcular_sobreposicao(boxA, boxB)

    # Evita divisão por zero caso alguma caixa seja inválida
    menor_area = min(area_boxA, area_boxB)
    if menor_area <= 0:
        return False, 0.0

    percentual_sobreposicao = area_intersecao / menor_area
    colisao = percentual_sobreposicao >= limiar

    return colisao, percentual_sobreposicao


# ============================================
# 4. CARREGA O MODELO
# ============================================

model = YOLO("yolo26n.pt")
# model = YOLO("path/to/best.pt")


# ============================================
# 5. EXECUTA A DETECÇÃO
# ============================================

results = model("https://ultralytics.com/images/bus.jpg")


# ============================================
# 6. PROCESSA OS RESULTADOS
# ============================================

for i, result in enumerate(results):
    print(f"\n================ RESULTADO {i} ================")

    boxes = result.boxes

    # Lista para armazenar apenas os carros detectados
    carros = []

    # --------------------------------------------
    # 6.1 Percorre todas as caixas detectadas
    # --------------------------------------------
    for j, box in enumerate(boxes):
        # Coordenadas da caixa no formato xyxy
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Coordenadas no formato xywh
        xc, yc, w, h = box.xywh[0].tolist()

        # Classe detectada
        cls_id = int(box.cls[0].item())
        cls_name = result.names[cls_id]

        # Confiança
        conf = float(box.conf[0].item())

        print(f"\nCaixa {j}:")
        print(f"  Classe: {cls_name} (id={cls_id})")
        print(f"  Confiança: {conf:.4f}")
        print(f"  xyxy: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
        print(f"  xywh: xc={xc:.2f}, yc={yc:.2f}, w={w:.2f}, h={h:.2f}")

        # --------------------------------------------
        # 6.2 Guarda apenas os objetos da classe "car"
        # --------------------------------------------
        if cls_name == "car":
            carros.append({
                "indice_original": j,
                "classe": cls_name,
                "confianca": conf,
                "box": [x1, y1, x2, y2],
                "geometria": criar_geometria_caixa([x1, y1, x2, y2])
            })

    # --------------------------------------------
    # 6.3 Exibe quantos carros foram encontrados
    # --------------------------------------------
    print(f"\nTotal de carros detectados: {len(carros)}")

    # Se houver menos de 2 carros, não há como comparar
    if len(carros) < 2:
        print("Não há carros suficientes para verificar colisão.")
        continue

    # --------------------------------------------
    # 6.4 Compara cada carro com todos os outros
    # --------------------------------------------
    print("\nVerificando colisões entre carros...")

    encontrou_colisao = False

    for a in range(len(carros)):
        for b in range(a + 1, len(carros)):
            carroA = carros[a]
            carroB = carros[b]

            boxA = carroA["box"]
            boxB = carroB["box"]

            colisao, percentual = ha_colisao(boxA, boxB, limiar=0.10)

            print(f"\nComparando carro {carroA['indice_original']} com carro {carroB['indice_original']}:")
            print(f"  Box A: {boxA}")
            print(f"  Box B: {boxB}")
            print(f"  Sobreposição relativa à menor caixa: {percentual * 100:.2f}%")

            if colisao:
                print("  ==> COLISÃO DETECTADA (>= 10% de sobreposição)")
                encontrou_colisao = True
            else:
                print("  ==> Sem colisão")

    if not encontrou_colisao:
        print("\nNenhuma colisão entre carros foi detectada.")