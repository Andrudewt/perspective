# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:59:47 2026

@author: wonky
"""

import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt


img = Image.open(r'./perspective.jpg')
w, h = img.size

fig, (asrc, anorm, afull) = plt.subplots(1, 3)

asrc.set_title('Исходное изображение')
anorm.set_title('Выбранная трапеция')
afull.set_title('Все изображение с учетом перспективы')

asrc.imshow(img)

# Координаты углов трапеции по картинке (тут получены вручную)
top_left = [1641, 281]
top_right = [2406, 277]
bottom_left = [1502, 1079]
bottom_right = [2751, 1061]
img_points = np.float32([top_left, top_right, bottom_left, bottom_right])

# Показать кординаты углов на картинке
asrc.plot(*img_points.T, marker='+', color='red', linewidth=0)

# координаты этих же точек на бумаге (*10, чтобы было видно картинку)
wmm = 500
hmm = 1000
real_pointst = np.float32([[0, 0], [wmm, 0], [0, hmm], [wmm, hmm]])

# Вычисление матрицы коррекции перспективы
pmatrix = cv.getPerspectiveTransform(img_points, real_pointst)

# выпрямленная картинка в пределах угла трапеции
cropped = cv.warpPerspective(np.array(img), pmatrix, (wmm, hmm))

# Показать выпрямленную картинку
anorm.imshow(cropped)

# Пересчет всей картинки (не только в пределах координат трапеции)

# Углы исходной каринки
corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
# Положение этих точек в реальном пространстве (применяем к ним матрицу коррекции перспективы)
t_corners = cv.perspectiveTransform(corners, pmatrix)
# Границы изображения в реальном пространстве
x_min, y_min = t_corners.min(axis=0).ravel()
x_max, y_max = t_corners.max(axis=0).ravel()
# Размеры области
new_w = int(np.ceil(x_max - x_min))
new_h = int(np.ceil(y_max - y_min))

# Матрица переноса (сдвиг картинки в положительную область
pmatrix_shift = np.array([[1, 0, -x_min],
                          [0, 1, -y_min],
                          [0, 0, 1]])
# Произведение матрицы переноса и матрицы перспективы
pmatrix_new = pmatrix_shift @ pmatrix

# полная картинка
corr = cv.warpPerspective(np.array(img), pmatrix_new, (new_w, new_h))

afull.imshow(corr)
plt.show()
