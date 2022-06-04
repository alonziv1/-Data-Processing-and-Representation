from re import A
import matplotlib.pyplot as plt
from numpy import arange
import numpy as np
import Bases
import plot_aux
from representations import representation
import functions 


a = 0
b= 1
intervals = 2**6
step = 1/intervals
x_values = arange(start=a, stop=b+step, step=step)
start = 0
end = abs(b-a)*intervals + 1 
n = 6

bases = [ Bases.Hadamard(n,a,b ), Bases.Walsh_Hadamard(n,a,b), Bases.Haar(n,a,b)]
names = ["Hadamard", "Walsh Hadamard", "Haar"]
chosen_k = []


for base, name in zip(bases, names):

    #display the base's functions
    values = base.h_values(x_values)
    # plot_aux.multipleGraphs(x_values, values, number_of_functions= base.size,Title= name, start= start, end= end)

    #display aproximation for the function in each base 
    current_representation = representation(functions.vector, base, a, b)
    represented_values = current_representation.all_values( x_values)
    plot_aux.SingleGraph(x_values, represented_values,Title= name + " representation", start= start, end= end, MSE = current_representation.optimalMSE())

    """#display the best k-term aproximtaion, k = 0,1,2,3

    size_1 = [[0],[1],[2],[3]]
    size_2 = [[0,1],[0,2],[0,3],[1,2],[1,3]]
    size_3 = [[0,1,2], [0,1,3],[1,2,3]]
    size_4 = [[0,1,2,3]]

    sizes = [size_1, size_2, size_3,size_4]

    original_values = [functions.phi(e) for e in x_values]
    for size in sizes:
        current_values = []
        MSE_list = []
        for k_list in size:
            current_representation = representation(functions.phi, base, a, b)
            current_values.append(current_representation.all_values( x_values, k_list = k_list))
            MSE_list.append(current_representation.optimalMSE(k_list))
        plot_aux.plotMulty(x_values,original_values , current_values,MSE_list, name, size)
    

    """



