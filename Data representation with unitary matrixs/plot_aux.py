import matplotlib.pyplot as plt

def multipleGraphs(X_values, Y_values, number_of_functions, Title, start, end):

    graph_counter = 0 
    cols = 4
    rows= 1+ int((number_of_functions-1)/cols)


    figure, axis = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):
            if graph_counter == number_of_functions:
                break

            axis[row, col].plot(X_values[start:end], Y_values[graph_counter][start:end], color='b',drawstyle='steps-post' )
            # axis[row, col].set_title(Title +': h_%d' %graph_counter)
            graph_counter +=1
            
    
    plt.show()

    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("x_values")
    plt.ylabel(ylabel)
    plt.title(Title)

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    plt.show()"""

