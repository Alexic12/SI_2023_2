"""from hubs.data_hub import data

Data = data()

Data.data_process('ETHEREUM_PRICE.xlsx')"""

from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron', file_name = 'OR.xlsx', iter = 100, alpha = 0.8, test_split = 0, norm = False, stop_condition = 0, neurons = 1)
