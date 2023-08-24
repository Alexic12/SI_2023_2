from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'PMC', file_name = 'Perceptron.XLSX', iter = 100, alfa = 1, test_split = 0, norm = False, stop_condition = 0, nfl = 5, neurons = 2)