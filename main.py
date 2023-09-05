"""from hubs.data_hub import data

Data = data()

Data.data_process('ETHEREUM_PRICE.xlsx')"""

from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron_multi', file_name = 'Cerveza.xlsx', iter = 500, alpha = 0.01, test_split = 0, norm = True, stop_condition = 0.1, neurons = 4, avoid_col = 0)
