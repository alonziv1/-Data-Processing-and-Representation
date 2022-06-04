from subprocess import NORMAL_PRIORITY_CLASS
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import numpy as np
import cmath 
import cmath
from scipy.linalg import dft

class Base:

    def __init__(self,n , a, b):

        if self.name == 'Fourier':
            self.size = n
        else:
            self.size = 2**n
        self.scaling_factor = (cmath.sqrt(self.size))
        self.scaling_factors = np.full(self.size,self.scaling_factor)
        self.a = a 
        self.b = b

    def h_k(self ,k, x_value):

        if x_value == self.b:
            interval = self.size - 1
        else:
            interval = int(((x_value + abs(self.a))/(abs(self.b-self.a))) * self.size)
        value = self.scaling_factors[k] * self.matrix[k][interval] 
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

    
    def norms(self):

        norms = []
        for row in self.matrix:

            norms.append(cmath.sqrt(sum([abs(e)**2 for e in row])))
        print("norms are:", norms)
    



class Hadamard(Base):

    def __init__(self,n,a,b):
        
        self.name = "Hadamard"
        super().__init__(n,a,b)
        self
        self.matrix = 1/(cmath.sqrt(self.size)) *self.__hadamard_matrix(n)


    def __hadamard_matrix (self, n):
        H_2= np.array([[1,1],[1,-1]])
        H_n = 1

        for i in range(n):
            H_n = np.kron(H_n,H_2)
        
        return H_n

  
class Walsh_Hadamard(Hadamard):

    def __init__(self, n,a,b):

        super().__init__(n,a,b)
        self.name = 'Walsh Hadamard'
        self.matrix = sorted(self.matrix,  key=self.__changesCounter)

        
    def __changesCounter(self,array):

        counter = 0
        for index in range(len(array)-1):
            if array[index] != array[index+1]:
                counter += 1
        return counter
        
class Haar(Base):

    def __init__(self,n,a,b):
        self.name = 'Haar'
        super().__init__(n,a,b)
        self.matrix = self.__haar_matrix(n)
        self.scaling_factors = self.__scaling_factors(n)

        
    def __haar_matrix (self, n):

        top = np.array([1,1])
        bottom = np.array([1,-1])
        size = 1
        H_n = 1

        for i in range(n):
            H_n = (1/cmath.sqrt(2))*(np.append(np.kron(H_n,top),np.kron(np.identity(size), bottom)))
            size *= 2
            H_n = np.reshape(H_n, newshape = (size,size))

        return H_n

    def __scaling_factors(self,n):

        root_2 = cmath.sqrt(2)
        scaling_factors = [root_2, root_2]
        for i in range(1,n):
            scaling_factors.extend(np.ones(2**i))
            scaling_factors = [e*root_2 for e in scaling_factors]

        return scaling_factors


class Fourier(Base):

    def __init__(self,n,a,b):
        self.name = 'Fourier'
        super().__init__(n,a,b)
        self.matrix = self.__DFT()

    def raiseToPower(self,array):
        powers  = np.arange(self.size)
        array = [base**power for base,power in zip(array, powers)]
        return array




    def __DFT(self):

        w = cmath.exp((-2j*cmath.pi)/self.size)
        matrix = np.full(self.size, fill_value= w )
        matrix = self.raiseToPower(matrix)
        matrix = [np.full(self.size, fill_value= base) for base in matrix]
        matrix = np.array([self.raiseToPower(row) for row in matrix])

        return matrix *  (1/(cmath.sqrt(self.size)))


"""my_dft  = Fourier(273,0,1)
auto_dft = dft(273,scale = 'sqrtn')

diff = np.ravel((my_dft.matrix -  auto_dft ))
diff = [abs(e) for e in diff]

print(max(diff))
"""
