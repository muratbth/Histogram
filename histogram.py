import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread('peppers.png', 0)  
# orjinal histogramı hesapla
original_hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# histogram eşitleme uygula
equalized_image = cv2.equalizeHist(image)

# eşitlenmiş histogramı hesapla
equalized_hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# Grafik Çizimi
plt.figure(figsize=(12, 6))

# Orijinal Görüntü ve Histogram
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Orijinal Görüntü')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(original_hist, color='blue')
plt.title('Orijinal Histogram')
plt.xlabel('Piksel Değerleri')
plt.ylabel('Piksel Sayısı')
plt.grid(True)

# Eşitlenmiş Görüntü ve Histogram
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Eşitlenmiş Görüntü')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(equalized_hist, color='red')
plt.title('Eşitlenmiş Histogram')
plt.xlabel('Piksel Değerleri')
plt.ylabel('Piksel Sayısı')
plt.grid(True)

plt.tight_layout()
plt.show()
