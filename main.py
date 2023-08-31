from hubs.neural_hub import Neural

N = Neural()

# N.run_model(model = 'perceptron_multi', file_name = 'ETHEREUM_PRICE.xlsx', iter = 100, alfa= 0.2, test_split = 0, norm=True, stop_condition=0.5, nfl = 2, neurons = 1)
N.run_model(model = 'perceptron_multi', file_name = 'train_heart.xlsx', iter = 40, alfa=0.2, test_split = 0.1, norm = True, stop_condition = 0, neurons = 1, avoid_col = 1)

##avoid column cuantas columnas quiero evitar al principio








