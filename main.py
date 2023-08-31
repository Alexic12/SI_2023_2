"""from hubs.data_hub import data

Data = data()

Data.data_process('ETHEREUM_PRICE.xlsx')"""

from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron_multi', file_name = 'OR.xlsx', iter = 3000, alpha = 0.7, test_split = 0, norm = False, stop_condition = 0.1, neurons = 1, avoid_col = 0)
