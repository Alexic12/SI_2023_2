from hubs.neural_hub import Neural


N = Neural()

model = 'xgb'
file_name = 'TOMA_DATOS_PID_2.xlsx'
iter = 2
alfa=0.2
test_split = 0.1
norm = True
stop_condition = 50
output = 1
avoid_col = 0
chk_name = 'PID_IDENT_1'
train = True
data_type = 'time_series'


N.run_model(model = model, file_name = file_name, iter = iter, alfa=alfa, test_split = test_split, norm = norm, stop_condition = stop_condition, neurons = output, avoid_col = avoid_col, chk_name = chk_name, train = train, data_type = data_type)

