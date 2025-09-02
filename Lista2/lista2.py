import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs("resultados", exist_ok=True)

def image_convolution(f, w, debug=False):
    N,M = f.shape
    n,m = w.shape
    a = int((n-1)/2)
    b = int((m-1)/2)
    w_flip = np.flip(np.flip(w,0),1)
    g = np.zeros(f.shape, dtype=np.uint8)
    for x in range(a,N-a):
        for y in range(b,M-b):
            sub_f = f[x-a:x+a+1, y-b:y+b+1]
            g[x,y] = np.sum(np.multiply(sub_f, w_flip)).astype(np.uint8)
    return g

img1 = cv2.imread("image.png")
img2 = cv2.imread("lena.png")
img1_pb = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
img2_pb = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

#Questão 1
w_med = np.ones((3,3))/9.0
img1_media = image_convolution(img1_pb, w_med)
img2_media = image_convolution(img2_pb, w_med)

cv2.imwrite("resultados/img1_media.png", img1_media)
cv2.imwrite("resultados/img2_media.png", img2_media)

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img1_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img1_media, cmap="gray"), plt.title("Média - image.png"), plt.axis("off")
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img2_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img2_media, cmap="gray"), plt.title("Média - lena.png"), plt.axis("off")
plt.show()

# Questão 2
def media_k(img, n=3, k=5):
    N,M = img.shape
    a = n//2
    g = np.zeros_like(img)
    for x in range(a,N-a):
        for y in range(a,M-a):
            v = img[x-a:x+a+1, y-a:y+a+1].flatten()
            v.sort()
            g[x,y] = np.mean(v[:k])
    return g

img1_mediak = media_k(img1_pb,3,5)
img2_mediak = media_k(img2_pb,3,5)

cv2.imwrite("resultados/img1_mediak.png", img1_mediak)
cv2.imwrite("resultados/img2_mediak.png", img2_mediak)

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img1_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img1_mediak, cmap="gray"), plt.title("Média k-vizinhos - image.png"), plt.axis("off")
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img2_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img2_mediak, cmap="gray"), plt.title("Média k-vizinhos - lena.png"), plt.axis("off")
plt.show()


#Questão 3
def mediana(img, n=3):
    N,M = img.shape
    a = n//2
    g = np.zeros_like(img)
    for x in range(a,N-a):
        for y in range(a,M-a):
            v = img[x-a:x+a+1, y-a:y+a+1].flatten()
            g[x,y] = np.median(v)
    return g

img1_mediana = mediana(img1_pb,3)
img2_mediana = mediana(img2_pb,3)

cv2.imwrite("resultados/img1_mediana.png", img1_mediana)
cv2.imwrite("resultados/img2_mediana.png", img2_mediana)

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img1_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img1_mediana, cmap="gray"), plt.title("Mediana - image.png"), plt.axis("off")
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img2_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img2_mediana, cmap="gray"), plt.title("Mediana - lena.png"), plt.axis("off")
plt.show()


#Questão 4
w_lap = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
img1_lap = image_convolution(img1_pb, w_lap)
img2_lap = image_convolution(img2_pb, w_lap)

cv2.imwrite("resultados/img1_lap.png", img1_lap)
cv2.imwrite("resultados/img2_lap.png", img2_lap)

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img1_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img1_lap, cmap="gray"), plt.title("Laplaciano - image.png"), plt.axis("off")
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img2_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img2_lap, cmap="gray"), plt.title("Laplaciano - lena.png"), plt.axis("off")
plt.show()

#Questão 5
def roberts(img):
    gx = np.array([[1,0],[0,-1]])
    gy = np.array([[0,1],[-1,0]])
    imgx = image_convolution(img, gx)
    imgy = image_convolution(img, gy)
    g = np.sqrt(imgx.astype(np.float32)**2 + imgy.astype(np.float32)**2)
    return np.clip(g,0,255).astype(np.uint8)

img1_roberts = roberts(img1_pb)
img2_roberts = roberts(img2_pb)

cv2.imwrite("resultados/img1_roberts.png", img1_roberts)
cv2.imwrite("resultados/img2_roberts.png", img2_roberts)

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img1_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img1_roberts, cmap="gray"), plt.title("Roberts - image.png"), plt.axis("off")
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img2_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img2_roberts, cmap="gray"), plt.title("Roberts - lena.png"), plt.axis("off")
plt.show()

#  Questão 6
gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
gy = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
img1_prewitt = np.sqrt(image_convolution(img1_pb,gx)**2 + image_convolution(img1_pb,gy)**2)
img2_prewitt = np.sqrt(image_convolution(img2_pb,gx)**2 + image_convolution(img2_pb,gy)**2)

cv2.imwrite("resultados/img1_prewitt.png", img1_prewitt)
cv2.imwrite("resultados/img2_prewitt.png", img2_prewitt)

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img1_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img1_prewitt, cmap="gray"), plt.title("Prewitt - image.png"), plt.axis("off")
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img2_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img2_prewitt, cmap="gray"), plt.title("Prewitt - lena.png"), plt.axis("off")
plt.show()

# Questão 7 
gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
img1_sobel = np.sqrt(image_convolution(img1_pb,gx)**2 + image_convolution(img1_pb,gy)**2)
img2_sobel = np.sqrt(image_convolution(img2_pb,gx)**2 + image_convolution(img2_pb,gy)**2)

cv2.imwrite("resultados/img1_sobel.png", img1_sobel)
cv2.imwrite("resultados/img2_sobel.png", img2_sobel)

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img1_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img1_sobel, cmap="gray"), plt.title("Sobel - image.png"), plt.axis("off")
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(121), plt.imshow(img2_pb, cmap="gray"), plt.title("Original"), plt.axis("off")
plt.subplot(122), plt.imshow(img2_sobel, cmap="gray"), plt.title("Sobel - lena.png"), plt.axis("off")
plt.show()
