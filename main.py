from hubs.neural_hub import Neural

N=Neural()
#se importa desde el hub, la clase Neural con los parametros(modelo,nombre archivo)
N.run_model(model='perceptron',file_name='ETHEREUM_PRICE.xlsx')


