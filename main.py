from hubs.neural_hub import Neural

N = Neural()

# configuracion de los parametros para correr casa metodo
model = 'ffm_tf'
file_name = 'ETHEREUM_PRICE.xlsx'
iter = 500
alfa=0.3
test_split = 0.1
norm = True
stop_condition = 5
output = 1 #neuronas de salida
avoid_col = 0
chk_name = 'ETH_1' ## nombre del checkpint que voy a hacer
train = False

N.run_model(model = model, file_name = file_name, iter = iter, alfa=alfa, test_split = test_split, norm = norm, stop_condition = stop_condition, neurons = output, avoid_col = avoid_col, chk_name = chk_name, train = train)