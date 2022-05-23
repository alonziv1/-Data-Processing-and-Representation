from re import A
import matplotlib.pyplot as plt
from numpy import arange
import numpy as np
import Bases
import plot_aux
from representations import representation
import functions 

step = 1/(2**6)
x_values = arange(start=0, stop=1, step=step)
start = 0
end = 2**6
n = 3


bases = [ Bases.Hadamard(n), Bases.Walsh_Hadamard(n), Bases.Haar(n)]
names = ["Hadamard", "Walsh Hadamard", "Haar"]
representations = []

for base, name in zip(bases, names):
    values = base.h_values(x_values)
    plot_aux.multipleGraphs(x_values, values, number_of_functions= base.size,Title= name, start= start, end= end)
    if name == "Haar":
        print(base.matrix)
    current_representation = representation(functions.constant_f_x, base, 0, 1)
    print(name , "- MSE is: " , current_representation.optimalMSE() , ", coefficients are: " , current_representation.coefficients )

    



