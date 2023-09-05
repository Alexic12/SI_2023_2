from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = "perceptron_multicapa" , file_name ="TRAIN.xlsx", iter = 20, alfa= 0.15, test_split = 0.1, norm = True, stop_condition = 0, neurons = 1, avoid_col = 0)
