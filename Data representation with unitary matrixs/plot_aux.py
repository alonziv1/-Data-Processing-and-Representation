import matplotlib.pyplot as plt

def multipleGraphs(X_values, Y_values, number_of_functions, Title, start, end):

    graph_counter = 0 
    cols = 4
    rows= 1+ int((number_of_functions-1)/cols)
    figure, axis = plt.subplots(rows,cols)
    if number_of_functions <=4:
        for col in range(cols):
            if graph_counter == number_of_functions:
                break
            axis[col].plot(X_values[start:end], Y_values[graph_counter][start:end], color='b',drawstyle='steps-post' )
            axis[col].set_title(Title +': ha_%d' %graph_counter)
            graph_counter +=1
    else:
        for row in range(rows):
            for col in range(cols):
                if graph_counter == number_of_functions:
                    break

                axis[row, col].plot(X_values[start:end], Y_values[graph_counter][start:end], color='b' )
                axis[row, col].set_title(Title +': ha_%d' %graph_counter)
                graph_counter +=1
            
    
    plt.show()

def SingleGraph(X_values, Y_values, Title, start, end,MSE):

    plt.plot(X_values[start:end], Y_values[start:end], color='g', label = Title  ) 
    plt.title(Title+", MSE is " +'{0:.3g}'.format(MSE))   
    plt.show()




def plotMulty(X_values,original_values ,Y_values,MSE_list, Base_name, size):

    colors = ['g', 'b', 'c', 'm', 'y']

    plt.plot(X_values,original_values, color='r', label= "original")


    for i, k_list in enumerate(size):
        # Plotting both the curves simultaneously
        plt.plot(X_values, Y_values[i], color=colors[i], label="functions" + str(k_list)+ ", MSE is " +'{0:.3g}'.format(MSE_list[i]))

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(Base_name + " with " + str(len(size[0])) +" functions")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    plt.show()

