import os
import matplotlib.pyplot as plt
import cv2
from cv2.typing import MatLike 

def transpose_matrix(matrix):
    if not isinstance(matrix[0], list):
        return [[element] for element in matrix]

    height = len(matrix)
    width =  len(matrix[0])
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
        return None

    result = []
    for y in range(height1):
        for x in range(width2):
            row = []
            total = 0
            for k in range(width1):
                total += matrix1[y][k] * matrix2[k][x]
            row.append(total)
        result.append(row)

    return result

def load_images_from_folder(path) -> list[MatLike]:
    result = []
    for file_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, file_name))
        if img is not None:
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result.append(img)

    return result

def vectorize_image(image:MatLike):
    result = []
    height, width, _ = image.shape
    
    for y in range(height):
        for x in range(width):
            result.append(image[y][x])

    return result

def vectorize_images(images):
    return [vectorize_image(image) for image in images]

def find_average_face_vector(vectorized_images):
    num_of_images = len(vectorized_images)
    num_of_pixels = len(vectorized_images[0])
    result = [[0.0, 0.0, 0.0] for _ in range(num_of_pixels)]

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
modified_image_vector = subtract_mean_face(vectorized_images, average_face_vector)
transposed_matrix = transpose_matrix(modified_image_vector)
covariance_matrix = multiply_matrix(modified_image_vector, transposed_matrix) # M*M matrix