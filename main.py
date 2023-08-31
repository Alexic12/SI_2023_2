from hubs.neural_hub import Neural

N = Neural ()

N.run_model(model = 'perceptron_multi', file_name = 'train1.xlsx', iter=40, alfa=0.15, test_split=0.1, norm=True, stop_condition=0,neurons=1, avoid_col=0)
