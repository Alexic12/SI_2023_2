from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron', file_name = 'OR.xlsx', iter = 50, alfa = 0.2, test_split=0)

