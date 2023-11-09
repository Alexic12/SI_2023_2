from hubs.neural_hub import Neural # Con esta linea llamamos a hubs.neural_hub e importamos todos la clase neural donde se encuntra los diferentes modelos

N = Neural()
model = 'conv_tf'
file_name = 'HEART_DISEASE_DB.xlsx'
iter = 20
alfa=0.1
test_split = 0
norm = True
stop_condition = 50
output = 1 # neuronas de salidas.
avoid_col = 0
chk_name = 'ETH_9'
train = True # esto es para definir si deseamos o no entrenar el modelo 


N.run_model(model = model, file_name = file_name, iter = iter, alfa=alfa, test_split = test_split, norm = norm, stop_condition = stop_condition, neurons = output, avoid_cols = avoid_col, chk_name = chk_name, train = train)


# model : aqui pones el nombre del modelo que deseamos utilizar por ahora tenemos el perceptron multi y el perceptron solito 
# file_name: Aqui es donde se pone el archivo.xlsx donde debe estar organzido y con porcesamiento de datos , recordar si el archvio llega como scv cambiarlo a Xls. 
#iter: aqui puedo modificar las iteraciones que deseo realizar una iteracion es cuando pasa por todos los pesos y es el ultimo dato que toca modificar para cuadrar.
#alfa:Esta es la tasa de aprendizaje se utiliza para ajustar los peso de la red en el proceso de entrenamiento y controla rapidez con la que la red neunoral converga hacia la solucion optima. entre mas little es el entrenmaneot puede ser mas lento y si es muy grande pude ser muy rapido y volverse inestable es el penultimo dato que toca modificar.
#test_split:Es el porcentaje de datos deseo partir para hacer pruebas. 
#norm:Es para normalizar los datos de entradas antes de las redes neuronales para evitar redundancia, datos erroneos, mejorar el rendimiento por la escala similiar y evitar que los datos no normalizados multiplequen los pesos y los vulevan muy grandes.
#stop_condition:Este nos indica si deseamos parar cuando el ecm este en cero !!!!Actualmente descactivado!!!
#nuerons aqui defino la cantidad de neuronas que deso tener de salida.
#avoid_cols: es para filtras informacion inecesario si tenemos en las primeras columnas como years o etc 
# El main mueve el neural hub y envia al modelo se peude definir el nombre.