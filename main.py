import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from cv2.typing import MatLike 

def transpose_matrix(matrix):
    if not matrix or not isinstance(matrix[0], list):
        return [[element] for element in matrix] if matrix else []
    
    height = len(matrix)
    width = len(matrix[0])
    transposed_matrix = []
    
    for x in range(width):
        row = []
        for y in range(height):
            row.append(matrix[y][x])
        transposed_matrix.append(row)
    
    return transposed_matrix

def multiply_matrix(matrix1, matrix2):
    height1 = len(matrix1)
    width1 = len(matrix1[0])
    height2 = len(matrix2)
    width2 = len(matrix2[0])

    if width1 != height2:
        raise ValueError("matrix1의 열 수와 matrix2의 행 수가 일치하지 않습니다.")

    result = []
    for y in range(height1):
        row = []
        for x in range(width2):
            total = 0
            for k in range(width1):
                total += matrix1[y][k] * matrix2[k][x]
            row.append(total)
        result.append(row)

    return result

def load_images_from_folder(path) -> list[MatLike]:
    result = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"폴더 '{path}'를 찾을 수 없습니다.")
    
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path)
            if img is not None:
                result.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return result

def vectorize_image(image):
    return [int(pixel) for row in image for pixel in row]

def vectorize_images(images):
    return [vectorize_image(image) for image in images]

def find_average_face_vector(vectorized_images):
    num_of_images = len(vectorized_images)
    num_of_pixels = len(vectorized_images[0])
    result = [0.0 for _ in range(num_of_pixels)]

    for image in vectorized_images:
        for i in range(num_of_pixels):
             result[i]+=image[i]

    for i in range(num_of_pixels):
        result[i] /= num_of_images

    return result

def subtract_mean_face(vectorized_images, avg_face_vector):
    num_of_pixels = len(vectorized_images[0])
    result = []

    for image in vectorized_images:
        result.append([image[i] - avg_face_vector[i] for i in range(num_of_pixels)])
    return vectorized_images

face_images = load_images_from_folder("Images")
vectorized_images = vectorize_images(face_images)
average_face_vector = find_average_face_vector(vectorized_images)
modified_image_vectors = subtract_mean_face(vectorized_images, average_face_vector) # matrix A
product_matrix = multiply_matrix(modified_image_vectors, transpose_matrix(modified_image_vectors)) # matrix L

eigenvalues, eigenvectors = np.linalg.eig(product_matrix) # 추후 따로 구현
eigen_pairs = list(zip(eigenvalues, eigenvectors))
eigen_pairs.sort(key=lambda x: x[0], reverse=True)
sorted_eigenvectors = []
for eigen_pair in eigen_pairs:
    sorted_eigenvectors.append(eigen_pair[1])

covariance_vectors = multiply_matrix(sorted_eigenvectors, modified_image_vectors) # matrix V (M x N)