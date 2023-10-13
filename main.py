from hubs.neural_hub import Neural

N = Neural()

model = 'xgb'
file_name = 'DATA_SENO_DIRECTO.xlsx'
iter = 2
alfa=0.7
test_split = 0.1 # El ffm no va a funcionar sin esto, no puede ser cero para ese modelo
norm = True
stop_condition = 50
output = 1
avoid_col = 0
chk_name = 'Seno_1'
train = True
data_type = 'time_series'

N.run_model(model = model, file_name = file_name, iter = iter, alfa=alfa, test_split = test_split, norm = norm, stop_condition = stop_condition, neurons = output, avoid_col = avoid_col, chk_name = chk_name, train = train, data_type = data_type)

#perceptron una capa
'''
wij(t+1) = wij(t) - alfa*decm/dwij

decm/dwij = decm/dyk * dyk/daj * daj/dwij

wij(t+1) = wij(t) + alfa*ek*xi
'''

#perceptron multicapa
'''
cjk(t+1) = cjk(t) - alfa*decm/dcjk

decm/dcjk = decm/dyk * dyk/dak * dak/dcjk

cjk(t+1) = cjk(t) + alfa*ek*yk*(1-yk)*hj


wij(t+1) = wij(t) - alfa*decm/dwij

decm/dwij = decm/dyk * dyk/dak * dak/dhj * dhj/daj * daj/wij

wij(t+1) = wij(t) + alfa*ek*yk*(1-yk)*cjk*hj*(1-hj)*xi
'''