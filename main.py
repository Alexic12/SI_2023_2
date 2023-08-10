from hubs import Data
from hubs.neural_hub import Neural
'''
data = Data()

Data.data_process('ETHEREUM_PRICE.xlsx')
'''

N = Neural()
N.run_model(model = 'percepton', file_name= 'ETHEREUM_PRICE.xlsx', iter = 100)

