"""from hubs.data_hub import data

Data = data()

Data.data_process('ETHEREUM_PRICE.xlsx')"""

from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron_multi', file_name = 'ETHEREUM_PRICE.xlsx', iter = 100, alpha = 0.8, test_split = 0, norm = True, stop_condition = 0.1, neurons = 1)
