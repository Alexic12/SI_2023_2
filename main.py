from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron', file_name = 'ETHEREUM_PRICE.XLSX', iter = 100, alfa = 0.2)