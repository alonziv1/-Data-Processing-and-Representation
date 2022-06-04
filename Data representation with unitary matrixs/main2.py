from re import A
import matplotlib.pyplot as plt
from numpy import arange
import numpy as np
import Bases
import plot_aux
from representations import representation
import functions 
import cmath

a = 0
b= 10
intervals = 2**4
step = 1/intervals
x_values = arange(start=a, stop=b+step, step=step)
start = 0
end = abs(b-a)*intervals + 1 
n = 4

base = Bases.Fourier(n,a,b)
name = "Fourier"

values = base.h_values(x_values)
# plot_aux.multipleGraphs(x_values, values, number_of_functions= base.size,Title= name, start= start, end= end)

#display aproximation for the function in each base 
current_representation = representation(cmath.cos, base, a, b)
represented_values = current_representation.all_values( x_values)
plot_aux.SingleGraph(x_values, represented_values,Title= name + " representation", start= start, end= end,  MSE = 0 )

#MSE = current_representation.optimalMSE()