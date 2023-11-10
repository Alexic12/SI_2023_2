from hubs.neural_hub import Neural

N = Neural()

model = 'xgb'
file_name = 'TOMA_DATOS_PID_DIR.XLSX'
iter = 2
alfa = 0.2
test_split = 0.1
norm = True
stop_condition = 50
nfl = 0 
output = 1
avoid_col = 0
chk_name='IDENT_PID_DIR'
train = True
data_type = 'time_series'
identificacion = 'directa' ##Por lo general para inversa se debe reducir ventana con respecto a directa
adapt = True

N.run_model(model = model, file_name = file_name, iter = iter, alfa = alfa, test_split = test_split, norm = norm, stop_condition = stop_condition, nfl = nfl , neurons = output, avoid_col = avoid_col, chk_name = chk_name, train = train, data_type = data_type, identificacion = identificacion, adapt = adapt)
