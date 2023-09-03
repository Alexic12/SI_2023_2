from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'perceptron_multi', file_name = 'Heart_Train.xlsx', iter = 20, alfa = 0.15, test_split=0.1 , norm = True, stop_condition=0, neurons=1, avoid_cols = 0)

