import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import numpy as np
import math 


class Base:

    def __init__(self,n):

        self.size = 2**n

    def h_k(self ,k, x_value):
        
        if x_value == 1:
            interval = self.size - 1
        else:
            interval = int(x_value * self.size)
        value = (self.size**0.5) * self.matrix[k][interval] 
        return value

    def h_k_values(self, k ,x_values):

        values = []
        for x_value in x_values:
            values.append(self.h_k(k,x_value))
        return values

    def h_values(self, x_values):

        values = []

        for k in range(self.size):
            values.append(self.h_k_values(k ,x_values))

        return values



class Hadamard(Base):

    def __init__(self,n):

        super().__init__(n)
        self.matrix = (1/math.sqrt(self.size))* self.__hadamard_matrix(n)


    def __hadamard_matrix (self, n):
        H_2= np.array([[1,1],[1,-1]])
        H_n = 1

        for i in range(n):
            H_n = np.kron(H_n,H_2)
        
        return H_n

  
class Walsh_Hadamard(Hadamard):

    def __init__(self, n):

        super().__init__(n)
        self.matrix = sorted(self.matrix,  key=self.__changesCounter)


    def __changesCounter(self,array):

        counter = 0
        for index in range(len(array)-1):
            if array[index] != array[index+1]:
                counter += 1
        return counter
        
class Haar(Base):

    def __init__(self,n):
        super().__init__(n)
        self.matrix = self.__haar_matrix(n)

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

    
        
  