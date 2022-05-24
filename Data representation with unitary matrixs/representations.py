import scipy.integrate as integrate
import plot_aux
from numpy import arange


class representation:

    def __init__(self, original_function, base, start, end):

        self.base = base
        self.original_function = original_function
        self.start = start
        self.end = end
        self.coefficients = self.__optimalCoefficients() 


    def represented_function(self, x_value, k_list):

        if k_list == None: 
            k_list = arange(len(self.coefficients))

        sum = 0 
        for k in k_list:
            sum += self.coefficients[k] * self.base.h_k(k, x_value)

        return sum

    def all_values(self, x_values, k_list = None ):

        values = []

        for x in x_values:

            values.append(self.represented_function(x, k_list))

        return values

    def integrand_values(self,k, x_values):

        values = []

        self.current_k = k 
        for x in x_values:

            values.append(self.__integrand(x))

        return values


    def optimalMSE(self, k_list = None):

        if k_list == None: 
            k_list = arange(len(self.coefficients))

        energy = integrate.quad(self.__sqrd_original_function, self.start, self.end)[0]/(abs(self.start-self.end))

        selected_coef = [self.coefficients[k] for k in k_list]
        coef_sum = sum([coef**2 for coef in selected_coef])

        return abs((energy - coef_sum)/((abs(self.start-self.end))**2))
 

    def __optimalCoefficients(self):
        
        coefficients = []
        for k in range(self.base.size):
            self.current_k = k
            coefficients.append(self.__innerProduct(self.start, self.end ) / abs(self.end- self.start))

        return coefficients


    def __integrand(self, x_value):
            return self.original_function(x_value)*self.base.h_k(self.current_k, x_value)


    def __innerProduct(self, start, end):

        inner_product = integrate.quad(self.__integrand, start,end ,limit= 10000 )

        # print("integral error is " ,inner_product[1])

        return inner_product[0]

    def __sqrd_original_function(self,x_value):

        return (self.original_function(x_value))**2




    
#test function

def f_x (x):
    return x**2

def some_h_k(k,x_value):
    return k
