import pandas as pd
import numpy as np


# Generar una matriz de 15x4 de números aleatorios entre 0 y 4

matrix = np.random.randint(1,5, size=(60, 4))
print(matrix)
matriz1 = pd.DataFrame(matrix, columns=['estado1', 'estado2', 'estado3', 'estado4'])


print(matriz1)

matriz1.to_csv('holacsv.csv', index = False)