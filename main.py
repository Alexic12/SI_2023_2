#from hubs.neural_hub import Neural

#N=Neural()
#se importa desde el hub, la clase Neural con los parametros(modelo,nombre archivo)
#N.run_model(model='perceptron',file_name='ETHEREUM_PRICE.xlsx',iter=100)

from hubs.neural_hub import Neural
N = Neural()
N.run_model(model = 'perceptron', file_name='TV_EjercicioClase.xlsx', iter = 100, alfa = 0.08, test_split=0,norm=False,stop_condition = 0,neurons=2)

