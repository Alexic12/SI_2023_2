from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron', file_name='TV_OR.xlsx', iter = 100, alfa = 0.2, test_split=0)

