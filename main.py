from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron', file_name = 'AND.xlsx', iter = 100, alfa=0.09, test_split = 0, norm=False, stop_condition = 0, neurons=2)

