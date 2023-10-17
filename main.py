from hubs.neural_hub import Neural


N = Neural()
#Configuracion general de los parametros de todas las redes neuronales para que funciones

model = 'xgb'
file_name = 'DATA_SENO_DIRECTO.xlsx'
iter = 2
alfa=0.7
test_split = 1
norm = True
stop_condition = 2
output = 1 #neuronas de salida
avoid_col = 0
chk_name = 'SENO_1' ## nombre del checkpint que voy a hacer
train = True
data_type = 'time_series'
identif = 'directa'



N.run_model(model = model, file_name = file_name, iter = iter, alfa=alfa, test_split = test_split, norm = norm, stop_condition = stop_condition, neurons = output, avoid_col = avoid_col, chk_name = chk_name, train = train, data_type = data_type)
