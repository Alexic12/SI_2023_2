"""from hubs.data_hub import data

Data = data()

Data.data_process('ETHEREUM_PRICE.xlsx')"""

from hubs.neural_hub import Neural

N = Neural()

model = 'ffm_tf'
file_name = 'Cardiaco.xlsx'
iter = 500
alpha = 0.001
test_split = 0.1
norm = True
stop_condition = 50
neurons = 1
avoid_col = 0
chk_name = 'ffm_tf_model_cardiaco'
train = True

N.run_model(model = model, file_name = file_name, iter = iter, alpha = alpha, test_split = test_split, norm = norm, stop_condition = stop_condition, neurons = neurons, avoid_col = avoid_col, chk_name = chk_name, train = train)
