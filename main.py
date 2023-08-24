from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron', file_name = 'DOS_SALIDAS.xlsx', iter = 50, alfa = 0.2, test_split=1, norm = False, stop_condition=0, n_ocultas=2, n_salidas=1)

