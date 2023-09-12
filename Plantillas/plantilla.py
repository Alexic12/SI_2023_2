#########***********#########
#METODO LARGO PARA CREAR MATRIZ ALEATORIA#
# import random

# import pandas as pd

# Verde = 1
# Amarillo = 2
# Cafe = 3
# VerdeClaro = 4

# Rojo = 1

# Bajo =1 
# Grueso = 2
# Alto = 3
# Corto = 4

# NoTiene = 1
# Pequeño = 2
# Mediano =3
# Grande = 4

# # Definimos las categorías y sus valores numéricos

# colores_hoja = [Verde, Amarillo, Cafe, VerdeClaro]

# colores_fruto = [Rojo, Amarillo, Verde, Cafe]

# tallos = [Bajo, Grueso, Alto, Corto]

# tamanios_fruto = [NoTiene, Pequeño, Mediano, Grande]

# # Crear una lista para almacenar los datos generados aleatoriamente

# data = []

# # Generar al menos 15 datos para cada categoría

# for i in range(15):

#     data.append([random.choice(colores_hoja),random.choice(colores_fruto),random.choice(tallos), random.choice(tamanios_fruto)])

# # Crear un DataFrame a partir de los datos generados
# df = pd.DataFrame(data, columns=["Color de la hoja", "Color del fruto", "Tallo", "Tamaño del fruto"])

# # Mostrar el DataFrame
# print(df)

# df.to_csv('BASE_DATOS.csv', index = False)



#########***********#########
#METODO CORTO PARA CREAR MATRIZ ALEATORIA# 
import pandas as pd
import numpy as np

# Generar una matriz de 15x4 de números aleatorios entre 0 y 4
matrix = np.random.randint(1,5, size=(60, 4))

print(matrix)

matriz1 = pd.DataFrame(matrix, columns=['colores_hoja', 'colores_fruto', 'tallos', 'tamanios_fruto'])

print(matriz1) 

matriz1.to_csv('BASE_DATOS.csv', index = False)

