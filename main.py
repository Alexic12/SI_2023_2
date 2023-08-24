from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron_mult', file_name = 'Perceptron.xlsx', iter = 100, alfa= 0.2, test_split = 0, norm=False, stop_condition=0, nfl = 2, neurons = 1)











