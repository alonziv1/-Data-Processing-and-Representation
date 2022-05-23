import scipy.integrate as integrate

class representation:

    def __init__(self, original_function, base, start, end):

        self.base = base
        self.original_function = original_function
        self.start = start
        self.end = end
        self.coefficients = self.__optimalCoefficients() 


    def represented_function(self, x_value):

        sum = 0 
        for k, coefficient in enumerate(self.coefficients):
            sum += coefficient * self.base.h_k(k, x_value)

        return sum


    def optimalMSE(self):

        energy = integrate.quad(self.__sqrd_original_function, self.start, self.end)[0]
        coef_sum = sum([coef**2 for coef in self.coefficients])

        return energy - coef_sum
 

    def __optimalCoefficients(self):
        
        coefficients = []
        for k in range(self.base.size):
            self.current_k = k
            coefficients.append(self.__innerProduct(self.start, self.end ))

        return coefficients


    def __integrand(self, x_value):
            return self.original_function(x_value)*self.base.h_k(self.current_k, x_value)


    def __innerProduct(self, start, end)->float:

        return integrate.quad(self.__integrand, start,end)[0]

    def __sqrd_original_function(self,x_value):

        return (self.original_function(x_value))**2




    
#test function

def f_x (x):
    return x**2

def some_h_k(k,x_value):
    return k
