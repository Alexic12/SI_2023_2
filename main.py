from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = "perceptron_multicapa" , file_name ="AND_OR.xlsx", iter = 20, alfa= 0.9, test_split = 0, norm = True, stop_condition = 0, outputs = 2, avoid_col = 0)
