from hubs.neural_hub import Neural

N = Neural()
#N..run_model(model ='perceptron', file_name ='compuesta.xlsx', iter=50, alfa=0.2, test_split = 0, norm=False, stop_condition=0, neurons=2)
#N.run_model(model ='perceptron_multi', file_name ='BASE_DATOS.xlsx', iter=100, alfa=0.01, test_split = 0, norm=True, stop_condition=0, outputs=4, avoid_col=0)
N.run_model(model = 'ffm_tf', file_name = 'train_heart.xlsx', iter = 50, alfa=0.002, test_split = 0.1, norm = True, stop_condition = 50, outputs=4, avoid_col = 0)