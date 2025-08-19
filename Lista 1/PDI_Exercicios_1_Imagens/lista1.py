import cv2
import numpy as np


# Exercício 1: Conversão para Níveis de Cinza

exe1 = cv2.imread('PDI_Exercicios_1_Imagens/imgAna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Niveis de Cinza - imgAna', exe1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('cinza.png', exe1)

exe1_1 = cv2.imread('PDI_Exercicios_1_Imagens/lena.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Niveis de Cinza - lena', exe1_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('cinza_1.png', exe1_1)


# Exercício 2: Negativo

img1 = cv2.imread("PDI_Exercicios_1_Imagens/imgAna.png")
img2 = cv2.imread("PDI_Exercicios_1_Imagens/lena.png")

exe2 = 255 - img1
cv2.imshow('Negativo - imgAna', exe2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('negativo.png', exe2)

exe2_1 = 255 - img2
cv2.imshow('Negativo - lena', exe2_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('negativo_1.png', exe2_1)


# Exercício 3

exe3 = cv2.normalize(img1, None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)
cv2.imshow('Normalizacao - imgAna', exe3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('normalizacao.png', exe3)

exe3_1 = cv2.normalize(img2, None, alpha=0, beta=100, norm_type=cv2.NORM_MINMAX)
cv2.imshow('Normalizacao - lena', exe3_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('normalizacao_1.png', exe3_1)


# Exercício 4

img_float1 = img1.astype(np.float32) / 255.0
exe4 = 40 * np.log1p(img_float1)
exe4 = cv2.normalize(exe4, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow('Logaritmo - imgAna', exe4)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('logaritmo.png', exe4)

img_float2 = img2.astype(np.float32) / 255.0
exe4_1 = 40 * np.log1p(img_float2)
exe4_1 = cv2.normalize(exe4_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow('Logaritmo - lena', exe4_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('logaritmo_1.png', exe4_1)


# Exercício 5

img_float1 = img1.astype(np.float32) / 255.0
exe5 = 2 * (img_float1 ** 2)
exe5 = cv2.normalize(exe5, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow('Potencia - imgAna', exe5)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('potencia.png', exe5)

img_float2 = img2.astype(np.float32) / 255.0
exe5_1 = 2 * (img_float2 ** 2)
exe5_1 = cv2.normalize(exe5_1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imshow('Potencia - lena', exe5_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('potencia_1.png', exe5_1)


# Exercício 6

img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
for i in range(8):
    bit = ((img_gray1 >> i) & 1) * 255
    cv2.imshow(f'Bit-plane {i} - imgAna', bit)
    cv2.imwrite(f'bit_imgAna_{i}.png', bit)
    cv2.waitKey(0)
cv2.destroyAllWindows()

img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
for i in range(8):
    bit = ((img_gray2 >> i) & 1) * 255
    cv2.imshow(f'Bit-plane {i} - lena', bit)
    cv2.imwrite(f'bit_lena_{i}.png', bit)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# Exercício 7

# (i) Histograma da imagem unequalized (em cinza)
img_gray3 = cv2.imread("PDI_Exercicios_1_Imagens/unequalized.jpg", cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([img_gray3],[0],None,[256],[0,256])
hist_img = np.zeros((300,256), dtype=np.uint8)
cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
for x,y in enumerate(hist):
    cv2.line(hist_img, (x,300), (x,300-int(y)), 255)
cv2.imshow("Histograma - unequalized", hist_img)
cv2.imwrite("Histograma_unequalized.png", hist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# (ii) Histogramas RGB da minha imagewm
for i,col in zip(range(3), ['b','g','r']):
    hist = cv2.calcHist([img1],[i],None,[256],[0,256])
    hist_img = np.zeros((300,256,3), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
    for x,y in enumerate(hist):
        cv2.line(hist_img, (x,300), (x,300-int(y)), (255 if col=='b' else 0,
                                                    255 if col=='g' else 0,
                                                    255 if col=='r' else 0))
    cv2.imshow(f"Histograma {col} - imgAna", hist_img)
    cv2.imwrite(f"Histograma_{col}_imgAna.png", hist_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# (iii) Histogramas A, B, C e D da miha imagem em cinza
img_gray_aluno = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# A - Histograma
hist = cv2.calcHist([img_gray_aluno],[0],None,[256],[0,256])
hist_img = np.zeros((300,256), dtype=np.uint8)
cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
for x,y in enumerate(hist):
    cv2.line(hist_img, (x,300), (x,300-int(y)), 255)
cv2.imshow("Histograma A - imgAna", hist_img)
cv2.imwrite("Histograma_A.png", hist_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# B - Histograma Normalizado
hist_norm = hist / hist.sum()
print("Histograma Normalizado - primeiros valores:", hist_norm[:10])

# C - Histograma Acumulado
hist_acum = np.cumsum(hist)
print("Histograma Acumulado - primeiros valores:", hist_acum[:10])

# D - Histograma Acumulado Normalizado
hist_acum_norm = hist_acum / hist_acum[-1]
print("Histograma Acumulado Normalizado - primeiros valores:", hist_acum_norm[:10])


# Exercício 8

img_eq1 = cv2.equalizeHist(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
cv2.imshow('Equalizacao - lena', img_eq1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('equalizacao_lena.png', img_eq1)

img_eq2 = cv2.equalizeHist(img_gray3)
cv2.imshow('Equalizacao - unequalized', img_eq2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('equalizacao_unequalized.png', img_eq2)

img_eq3 = cv2.equalizeHist(img_gray_aluno)
cv2.imshow('Equalizacao - imgAna', img_eq3)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('equalizacao_imgAna.png', img_eq3)
