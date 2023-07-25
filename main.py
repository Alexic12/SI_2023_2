"""from hubs.data_hub import data

Data = data()

Data.data_process('ETHEREUM_PRICE.xlsx')"""

from hubs.neural_hub import Neural

N = Neural

N.run_model(model = 'perceptron', file_name = 'ETHEREUM_PRICE.xlsx')
