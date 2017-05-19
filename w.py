import numpy as np
precTable=np.load('precTable.npy')
#print precTable
'''
print len(precTable)
print precTable.shape
print precTable.size
precTable=precTable.reshape(precTable,precTable.size)
np.savetxt(f, precTable, delimiter=',', fmt='%d')
'''

with open('precTable.csv','wb') as f:
    for row in precTable:
    	for xyz in row:
    		np.savetxt(f, xyz, delimiter=',', fmt='%d')

precTable=np.loadtxt('precTable.csv')
print precTable
