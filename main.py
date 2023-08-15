from hubs.neural_hub import Neural

N = Neural()

N.run_model(model='perceptron', file_name='basedatos_or.xlsx', iter=100, alfa=0.01, test_split=0, norm = False)
