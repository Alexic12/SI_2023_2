from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'PMC', file_name = 'ETHEREUM_PRICE.XLSX', iter = 100, alfa = 0.2, test_split = 0, norm = True, stop_condition = 1, nfl = 6, neurons = 1)