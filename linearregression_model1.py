import numpy as np

data = np.genfromtxt('team_score_data.csv',delimiter = ',',skip_header = 1, usecols=(0, 1, 2))

#print (data)

Out = np.genfromtxt('team_score_data.csv',delimiter = ',',skip_header = 1, usecols=(3))

#print(Out)


m=data.shape[0]
c=data.shape[1]
#print (m)

w = np.zeros(c)
#print(w)
b=0

for i in range (m):
    
    pred = np.dot(data,w) + b
    
    eror = pred - Out
    grad=np.dot(data.T,eror) / m
    w = w-0.01 * grad
    b= b-0.01 * np.mean(eror)
    
    print(w,b)


