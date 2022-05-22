import numpy as np
from scipy.linalg import hadamard

A = np.array([[1,1],[1,-1]])

my_hadamard_4 = np.kron(A, A)
my_hadamard_8 = np.kron(my_hadamard_4, A)
my_hadamard_16 = np.kron(my_hadamard_8, A)

# print(my_hadamard_16)

def __hadamard_matrix ( n):
        H_2= np.array([[1,1],[1,-1]])
        H_n = 1

        for i in range(n):
            H_n = np.kron(H_n,H_2)

        return H_n


# print(__hadamard_matrix(3))
# print(np.equal(my_hadamard_16,hadamard(16)))


def __haar_matrix (n):
    
    H_n = np.array([[1,1],[1,-1]])
    top = np.array([1,1])
    bottom = np.array([1,-1])
    size = 2
    
    for i in range(1,n):
        
        H_n = np.append(np.kron(H_n,top),np.kron(np.identity(size), bottom))
        size *= 2
        H_n = np.reshape(H_n, newshape = (size,size))
    
    return H_n