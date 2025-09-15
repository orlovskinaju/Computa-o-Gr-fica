import cv2
import numpy as np
import os

def exercicio1():
    img = cv2.imread("circuito.tif", cv2.IMREAD_GRAYSCALE)
    for i in range(1, 4):
        img = cv2.medianBlur(img, 3)
        cv2.imwrite(f"resultados/circuito_mediana_{i}.png", img)
    cv2.imshow("Ex1 - Filtro Mediana (3x)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exercicio2():
    img = cv2.imread("pontos.png", cv2.IMREAD_GRAYSCALE)

    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    filtrada = cv2.filter2D(img, -1, kernel)
    _, binaria = cv2.threshold(filtrada, 200, 255, cv2.THRESH_BINARY)

    cv2.imwrite("resultados/pontos_detectados.png", binaria)
    cv2.imshow("Ex2 - Pontos Isolados", binaria)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def exercicio3():
    img = cv2.imread("linhas.png", cv2.IMREAD_GRAYSCALE)

    h = np.array([[-1, -1, -1],
                  [ 2,  2,  2],
                  [-1, -1, -1]])
    
    v = np.array([[-1,  2, -1],
                  [-1,  2, -1],
                  [-1,  2, -1]])
    
    d1 = np.array([[ 2, -1, -1],
                   [-1,  2, -1],
                   [-1, -1,  2]])
    
    d2 = np.array([[-1, -1,  2],
                   [-1,  2, -1],
                   [ 2, -1, -1]])

    filtros = {"horizontal": h, "vertical": v, "diag45": d1, "diag-45": d2}
    resultados = []

    for nome, k in filtros.items():
        resp = cv2.filter2D(img, -1, k)
        _, binaria = cv2.threshold(resp, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"resultados/linhas_{nome}.png", binaria)
        resultados.append(binaria)

    combinado = resultados[0]
    for r in resultados[1:]:
        combinado = cv2.bitwise_or(combinado, r)

    cv2.imwrite("resultados/linhas_combinadas.png", combinado)
    cv2.imshow("Ex3 - Linhas Combinadas", combinado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def exercicio4():
    img = cv2.imread("igreja.png", cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite("resultados/igreja_canny.png", edges)
    cv2.imshow("Ex4 - Canny", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def region_grow(img, seed, threshold=5):
    h, w = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)
    visited = np.zeros_like(img, dtype=bool)

    x0, y0 = seed
    region_value = int(img[y0, x0])
    stack = [(x0, y0)]

    while stack:
        x, y = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True

        if abs(int(img[y, x]) - region_value) < threshold:
            mask[y, x] = 255
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        stack.append((nx, ny))
    return mask


def exercicio5():
    img = cv2.imread("root.jpg", cv2.IMREAD_GRAYSCALE)
    seed = (100, 150) 
    mask = region_grow(img, seed, threshold=15)
    cv2.imwrite("resultados/root_region.png", mask)
    cv2.imshow("Ex5 - Crescimento de Regiao", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def exercicio6():
    imgs = ["harewood.jpg", "nuts.jpg", "snow.jpg", "image.png"]
    for nome in imgs:
        if not os.path.exists(nome):
            print(f"[AVISO] Arquivo {nome} não encontrado, pulando...")
            continue
        img = cv2.imread(nome, cv2.IMREAD_GRAYSCALE)
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        saida = "resultados/" + os.path.splitext(nome)[0] + "_otsu.png"
        cv2.imwrite(saida, otsu)
        cv2.imshow(f"Ex6 - Otsu {nome}", otsu)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


exercicio1()
exercicio2()
exercicio3()
exercicio4()
exercicio5()
exercicio6()

