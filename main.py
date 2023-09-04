from hubs.neural_hub import Neural

N = Neural()

N.run_model(model='perceptron_multi', file_name='HeartAttack.xlsx', iter=10, alfa=0.1, test_split=0, norm = True, stop_condition=0, neurons=1)
