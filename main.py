from hubs.neural_hub import Neural

N = Neural()

N.run_model(model = 'ffm_tf', file_name = 'HEART_DISEASE_DB.xlsx', iter = 500, alfa=0.002, test_split = 0.1, norm = True, stop_condition = 50, neurons = 1, avoid_cols = 0)



