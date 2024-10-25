from copy import deepcopy as cp

a = [[1,1],1,1]
b = [cp(a),cp(a),cp(a)]
b[0][0][0]=0
print(b)