"""from hubs.data_hub import data

Data = data()

Data.data_process('ETHEREUM_PRICE.xlsx')"""

from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'ffm_tf', file_name = 'Cardiaco.xlsx', iter = 500, alpha = 0.002, test_split = 0.1, norm = True, stop_condition = 50, neurons = 1, avoid_col = 0)
