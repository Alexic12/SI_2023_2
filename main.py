"""from hubs.data_hub import data

Data = data()

Data.data_process('ETHEREUM_PRICE.xlsx')"""

from hubs.neural_hub import Neural

N = Neural()

model = 'xgb'
file_name = 'TOMA_DATOs_PID_DIR.xlsx'
iter = 1000
alpha = 0.05
test_split = 0.1
norm = True
stop_condition = 50
neurons = 1
avoid_col = 0
chk_name = 'INDENT_DIR_PID'
train = True
data_type = 'time_series'
iden = 'Directo'
windows_size = 3
horizon_size = 1

N.run_model(model = model, file_name = file_name, iter = iter, alpha = alpha, test_split = test_split, norm = norm, stop_condition = stop_condition, neurons = neurons, avoid_col = avoid_col, chk_name = chk_name, train = train, data_type = data_type, iden = iden, windows_size = windows_size, horizon_size = horizon_size)
