from re import A
import matplotlib.pyplot as plt
from numpy import arange
import Bases

import plot_aux

step = 1/(2**6)
x_values = arange(start=0, stop=1, step=step)
start = 0
end = (2**6)

def getBaseVectors(Base , x_values, n ):

    values = []

    for index in range(n):
        current_values = []
        for x_value in x_values:
            current_values.append(Bases.Vectors.getVector(Base,index,x_value))
        values.append(current_values)

    return values


Hadamard_Base = Bases.Hadamard(n = 3)
Hadamard_Values = getBaseVectors(Hadamard_Base, x_values, Hadamard_Base.size )
# plot_aux.multipleGraphs(x_values, Hadamard_Values, number_of_functions=Hadamard_Base.size,Title="Hadamard", start= start, end= end)

Walsh_Hadamard_Base = Bases.Walsh_Hadamard(Hadamard_Base.matrix)
Walsh_Hadamard_Values = getBaseVectors(Walsh_Hadamard_Base, x_values, Walsh_Hadamard_Base.size )
# plot_aux.multipleGraphs(x_values, Walsh_Hadamard_Values, number_of_functions=Walsh_Hadamard_Base.size,Title="Walsh Hadamard", start= start, end= end)

Haar_Base = Bases.Haar(n = 3)
Haar_Values = getBaseVectors(Haar_Base, x_values, Haar_Base.size )
plot_aux.multipleGraphs(x_values, Haar_Values, number_of_functions=Haar_Base.size,Title="Haar", start= start, end= end)

