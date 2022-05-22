import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import numpy as np
import math 

class Vectors:

    def getVector(base,index, x_value):
        
        if x_value == 1:
            interval = base.size - 1
        else:
            interval = int(x_value * base.size)
        value = (base.size**0.5) * base.matrix[index][interval] 
        return value



class Hadamard:

    def __hadamard_matrix (self, n):
        H_2= np.array([[1,1],[1,-1]])
        H_n = 1

        for i in range(n):
            H_n = np.kron(H_n,H_2)
        
        return H_n

    def __init__(self,n):
        self.size = 2**n
        self.matrix = (1/math.sqrt(self.size))* self.__hadamard_matrix(n)

    
class Walsh_Hadamard:

    def __changesCounter(self,array):

        counter = 0
        for index in range(len(array)-1):
            if array[index] != array[index+1]:
                counter += 1
        return counter
        
    def __init__(self, hadamard_matrix):
        self.size = len(hadamard_matrix)
        self.matrix = sorted(hadamard_matrix, key=self.__changesCounter)

class Haar:

    def __haar_matrix (self, n):
    
        
        top = np.array([1,1])
        bottom = np.array([1,-1])
        size = 2
        H_n = (1/math.sqrt(size))*np.array([[1,1],[1,-1]])
        
        for i in range(1,n):
            
            H_n = (1/math.sqrt(size))*np.append(np.kron(H_n,top),np.kron(np.identity(size), bottom))
            size *= 2
            H_n = np.reshape(H_n, newshape = (size,size))


        
        return H_n

    def __init__(self,n):
        self.size = 2**n
        self.matrix = self.__haar_matrix(n)
        
  