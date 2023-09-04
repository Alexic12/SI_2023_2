from hubs.neural_hub import Neural

N = Neural()

N.run_model(
    model = 'perceptron_multi', 
    file_name= 'HEART_DESEASE_PREDICTION.xlsx', 
    iter = 20, 
    alfa = 0.2, 
    test_split= 0, 
    norm = True, 
    stop_condition = 0,
    neurons = 1,
    avoid_col = 0
)