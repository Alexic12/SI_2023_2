from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron_multi', file_name = 'ETHEREUM_PRICE.xlsx', iter = 100, alfa=0.2, test_split = 0.1, norm = True, stop_condition = 0, neurons = 1)

