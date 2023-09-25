from hubs.neural_hub import Neural

N = Neural()
model = "xgb"
file_name ="HEART_DISEASE_DB .xlsx"
iter = 500
alfa= 0.02
test_split = 0.1
norm = True
stop_condition = 50
outputs = 1
avoid_col = 0
chk_name = "CP_1"
train= False

N.run_model(model=model,file_name=file_name,iter=iter,alfa=alfa,test_split=test_split,norm=norm,stop_condition=stop_condition,outputs=outputs,avoid_col=avoid_col,chk_name=chk_name,train=train)
