"""from hubs.data_hub import data

Data = data()

Data.data_process('ETHEREUM_PRICE.xlsx')"""

from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron', file_name = 'OR.xlsx', iter = 1000, alpha = 0.000001, split = 0)
