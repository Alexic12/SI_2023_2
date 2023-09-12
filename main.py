from hubs.neural_hub import Neural

N = Neural()

#N.run_model(model = 'PMC', file_name = 'Heart_Disease.XLSX', iter = 20, alfa = 0.02, test_split = 0, norm = True, stop_condition = 0, nfl = 14, neurons = 1, avoid_col = 0)
N.run_model(model = 'ffm_tf', file_name = 'Heart_Disease.XLSX', iter = 500, alfa = 0.002, test_split = 0.1, norm = True, stop_condition = 50, nfl = 0 , neurons = 1, avoid_col = 0)