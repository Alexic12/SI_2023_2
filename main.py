from hubs.neural_hub import Neural

N = Neural()

model = 'conv_tf'
file_name = 'Heart_Disease.XLSX'
iter = 2
alfa = 0.05
test_split = 0
norm = True
stop_condition = 50
nfl = 0 
output = 1
avoid_col = 0
chk_name='HD'
train = False

N.run_model(model = model, file_name = file_name, iter = iter, alfa = alfa, test_split = test_split, norm = norm, stop_condition = stop_condition, nfl = nfl , neurons = output, avoid_col = avoid_col, chk_name = chk_name, train = train)