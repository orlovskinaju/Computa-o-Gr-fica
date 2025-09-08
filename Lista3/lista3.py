import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

img1 = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("teste.tif", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("arara.png", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread("barra1.png", cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread("barra2.png", cv2.IMREAD_GRAYSCALE)
img6 = cv2.imread("barra3.png", cv2.IMREAD_GRAYSCALE)
img7 = cv2.imread("barra4.png", cv2.IMREAD_GRAYSCALE)


def espectro(img):
    F = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    Fshift = np.fft.fftshift(F)
    magnitude = cv2.magnitude(Fshift[:,:,0], Fshift[:,:,1])
    return 20*np.log(1+magnitude)   

esp1 = espectro(img1)
esp2 = espectro(img2)
esp3 = espectro(img3)
esp4 = espectro(img4)
esp5 = espectro(img5)
esp6 = espectro(img6)
esp7 = espectro(img7)


cv2.imwrite("resultados/img1_fourier.png", esp1)
plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img1, cmap="gray"), plt.title("Original - image.png"), plt.axis("off")
plt.subplot(122), plt.imshow(esp1, cmap="gray"), plt.title("Espectro - image.png"), plt.axis("off")
plt.show()


cv2.imwrite("resultados/img2_fourier.png", esp2)
plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img2, cmap="gray"), plt.title("Original - teste.tif"), plt.axis("off")
plt.subplot(122), plt.imshow(esp2, cmap="gray"), plt.title("Espectro - teste.tif"), plt.axis("off")
plt.show()


cv2.imwrite("resultados/img3_fourier.png", esp3)
plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img3, cmap="gray"), plt.title("Original - arara.png"), plt.axis("off")
plt.subplot(122), plt.imshow(esp3, cmap="gray"), plt.title("Espectro - arara.png"), plt.axis("off")
plt.show()


cv2.imwrite("resultados/img4_fourier.png", esp4)
plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img4, cmap="gray"), plt.title("Original - barra1.png"), plt.axis("off")
plt.subplot(122), plt.imshow(esp4, cmap="gray"), plt.title("Espectro - barra1.png"), plt.axis("off")
plt.show()


cv2.imwrite("resultados/img5_fourier.png", esp5)
plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img5, cmap="gray"), plt.title("Original - barra2.png"), plt.axis("off")
plt.subplot(122), plt.imshow(esp5, cmap="gray"), plt.title("Espectro - barra2.png"), plt.axis("off")
plt.show()


cv2.imwrite("resultados/img6_fourier.png", esp6)
plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img6, cmap="gray"), plt.title("Original - barra3.png"), plt.axis("off")
plt.subplot(122), plt.imshow(esp6, cmap="gray"), plt.title("Espectro - barra3.png"), plt.axis("off")
plt.show()


cv2.imwrite("resultados/img7_fourier.png", esp7)
plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img7, cmap="gray"), plt.title("Original - barra4.png"), plt.axis("off")
plt.subplot(122), plt.imshow(esp7, cmap="gray"), plt.title("Espectro - barra4.png"), plt.axis("off")
plt.show()

def aplica_filtro(img, D0, tipo="low"):
    # Transformada
    F = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    Fshift = np.fft.fftshift(F)

    # Filtro gaussiano
    h = cv2.getGaussianKernel(img.shape[0], D0)
    g = cv2.getGaussianKernel(img.shape[1], D0)
    H = h @ g.T
    if tipo == "high":
        H = 1 - H
    H = np.repeat(H[:, :, np.newaxis], 2, axis=2)

    # Aplica filtro
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    img_filtrada = cv2.idft(G)
    img_filtrada = cv2.magnitude(img_filtrada[:,:,0], img_filtrada[:,:,1])
    return cv2.normalize(img_filtrada, None, 0, 255, cv2.NORM_MINMAX)

img_aluno = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
teste = cv2.imread("teste.tif", cv2.IMREAD_GRAYSCALE)

for nome, img in [("teste", teste), ("image", img_aluno)]:
    low = aplica_filtro(img, 30, "low")
    high = aplica_filtro(img, 30, "high")

    cv2.imwrite(f"resultados/{nome}_low.png", low)
    cv2.imwrite(f"resultados/{nome}_high.png", high)

    plt.figure(figsize=(12,6))
    plt.subplot(131), plt.imshow(img, cmap="gray"), plt.title(f"Original - {nome}"), plt.axis("off")
    plt.subplot(132), plt.imshow(low, cmap="gray"), plt.title("Passa-baixa"), plt.axis("off")
    plt.subplot(133), plt.imshow(high, cmap="gray"), plt.title("Passa-alta"), plt.axis("off")
    plt.show()

    arara = cv2.imread("arara.png", cv2.IMREAD_GRAYSCALE)
filtro = cv2.imread("arara_filtro.png", cv2.IMREAD_GRAYSCALE)

F = cv2.dft(np.float32(arara), flags=cv2.DFT_COMPLEX_OUTPUT)
Fshift = np.fft.fftshift(F)

# Normaliza filtro (0-1) e aplica nos 2 canais
H = filtro / 255.0
H = np.repeat(H[:, :, np.newaxis], 2, axis=2)

Gshift = Fshift * H
G = np.fft.ifftshift(Gshift)
arara_filtrada = cv2.idft(G)
arara_filtrada = cv2.magnitude(arara_filtrada[:,:,0], arara_filtrada[:,:,1])
arara_filtrada = cv2.normalize(arara_filtrada, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite("resultados/arara_rejeita_banda.png", arara_filtrada)

plt.figure(figsize=(12,6))
plt.subplot(131), plt.imshow(arara, cmap="gray"), plt.title("Original - arara.png"), plt.axis("off")
plt.subplot(132), plt.imshow(filtro, cmap="gray"), plt.title("Filtro rejeita-banda"), plt.axis("off")
plt.subplot(133), plt.imshow(arara_filtrada, cmap="gray"), plt.title("Resultado filtrado"), plt.axis("off")
plt.show()
def cria_filtro(shape, R1, R2, tipo="passa-banda"):
    """
    Cria filtro circular passa-banda ou rejeita-banda
    shape: (linhas, colunas)
    R1: raio interno
    R2: raio externo
    tipo: "passa-banda" ou "rejeita-banda"
    """
    P, Q = shape
    u = np.arange(P) - P//2
    v = np.arange(Q) - Q//2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)

    if tipo == "passa-banda":
        H = np.logical_and(D >= R1, D <= R2).astype(np.float32)
    else:  # rejeita-banda
        H = np.logical_or(D < R1, D > R2).astype(np.float32)

    return np.repeat(H[:, :, np.newaxis], 2, axis=2)

def aplica_filtro(img, H):
    """
    Aplica um filtro H no domínio da frequência
    """
    F = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    Fshift = np.fft.fftshift(F)

    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    img_filtrada = cv2.idft(G)
    img_filtrada = cv2.magnitude(img_filtrada[:,:,0], img_filtrada[:,:,1])
    return cv2.normalize(img_filtrada, None, 0, 255, cv2.NORM_MINMAX)

# Carregar imagens
img_aluno = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
teste = cv2.imread("teste.tif", cv2.IMREAD_GRAYSCALE)

# Parâmetros dos filtros
R1, R2 = 20, 60

for nome, img in [("teste", teste), ("image", img_aluno)]:
    H_passa = cria_filtro(img.shape, R1, R2, "passa-banda")
    H_rejeita = cria_filtro(img.shape, R1, R2, "rejeita-banda")

    passa = aplica_filtro(img, H_passa)
    rejeita = aplica_filtro(img, H_rejeita)

    cv2.imwrite(f"resultados/{nome}_passa_banda.png", passa)
    cv2.imwrite(f"resultados/{nome}_rejeita_banda.png", rejeita)

    plt.figure(figsize=(14,6))
    plt.subplot(131), plt.imshow(img, cmap="gray"), plt.title(f"Original - {nome}"), plt.axis("off")
    plt.subplot(132), plt.imshow(passa, cmap="gray"), plt.title("Passa-banda"), plt.axis("off")
    plt.subplot(133), plt.imshow(rejeita, cmap="gray"), plt.title("Rejeita-banda"), plt.axis("off")
    plt.show()