import cv2
import numpy as np
import os

os.makedirs("resultados", exist_ok=True)

def exercicio1():
    img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

    se_a = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    se_b = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)

    erosao_a = cv2.erode(img, se_a)
    erosao_b = cv2.erode(img, se_b)
    dilat_a  = cv2.dilate(img, se_a)
    dilat_b  = cv2.dilate(img, se_b)

    cv2.imwrite("resultados/ex1_erosao_a.png", erosao_a)
    cv2.imwrite("resultados/ex1_erosao_b.png", erosao_b)
    cv2.imwrite("resultados/ex1_dilatacao_a.png", dilat_a)
    cv2.imwrite("resultados/ex1_dilatacao_b.png", dilat_b)

    cv2.imshow("Ex1 - Erosao/Dilatacao", dilat_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exercicio2():
    img = cv2.imread("quadrados.png", cv2.IMREAD_GRAYSCALE)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))  # maior que 40 e menor que 60
    erodida = cv2.erode(img, se)
    restaurada = cv2.dilate(erodida, se)

    cv2.imwrite("resultados/ex2_erosao.png", erodida)
    cv2.imwrite("resultados/ex2_restaurada.png", restaurada)
    cv2.imshow("Ex2 - Quadrados Restaurados", restaurada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exercicio3():
    img = cv2.imread("ruidos.png", cv2.IMREAD_GRAYSCALE)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    abertura = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    fechamento = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

    cv2.imwrite("resultados/ex3_abertura.png", abertura)
    cv2.imwrite("resultados/ex3_fechamento.png", fechamento)
    cv2.imshow("Ex3 - Fechamento", fechamento)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exercicio4():
    img = cv2.imread("cachorro.png", cv2.IMREAD_GRAYSCALE)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    borda_interna = img - cv2.erode(img, se)
    borda_externa = cv2.dilate(img, se) - img

    cv2.imwrite("resultados/ex4_borda_interna.png", borda_interna)
    cv2.imwrite("resultados/ex4_borda_externa.png", borda_externa)
    cv2.imshow("Ex4 - Borda Externa", borda_externa)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exercicio5():
    img = cv2.imread("gato.png", cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    seed_point = (10, 10)  # ajuste conforme a imagem
    flood = bin_img.copy()
    h, w = bin_img.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, mask, seed_point, 255)
    preenchido = cv2.bitwise_not(flood)

    cv2.imwrite("resultados/ex5_preenchimento.png", preenchido)
    cv2.imshow("Ex5 - Preenchimento", preenchido)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exercicio6():
    img = cv2.imread("quadrados.png", cv2.IMREAD_GRAYSCALE)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)

    if num_labels > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.zeros_like(img)
        mask[labels == idx] = 255
        color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        color[labels == idx] = (0,255,255)  # amarelo

        cv2.imwrite("resultados/ex6_componente_amarelo.png", color)
        cv2.imshow("Ex6 - Componente 80px", color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def exercicio7():
    img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil = cv2.dilate(img, se)
    ero = cv2.erode(img, se)
    grad = cv2.subtract(dil, ero)

    cv2.imwrite("resultados/ex7_dilatacao.png", dil)
    cv2.imwrite("resultados/ex7_erosao.png", ero)
    cv2.imwrite("resultados/ex7_gradiente.png", grad)
    cv2.imshow("Ex7 - Gradiente", grad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Chamadas
exercicio1()
exercicio2()
exercicio3()
exercicio4()
exercicio5()
exercicio6()
exercicio7()
