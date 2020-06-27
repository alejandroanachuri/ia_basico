import matplotlib.pyplot as plt
import numpy as np

def clases(x, y):
    class_1 = []
    class_2 = []
    for i in range(len(y)):
        if y[i] < 0:
           class_2.append(x[i])
        else:
            class_1.append(x[i])
    return class_1, class_2

def perceptron(x_set, y_set ):
    print ("Procesando perceptron....")
    use_bias = True
    pesos = np.array([0.0, 0.0])
    bias = 0.0
    n_epochs = 1000
    error = True
    iteraciones = 1
    secuencia = [[0.0,0.0]]
    cant_errores = 0
    while error or n_epochs < 1000:
        print ("Epoca ",iteraciones)
        iteraciones = iteraciones + 1
        error = False
        for j in range(len(x_set)):
            pe = (x_set[j][0]*pesos[0]+x_set[j][1]*pesos[1])
            if use_bias:
                pe = pe+bias
            if pe*y_set[j] <= 0:
                pesos[0] = pesos[0] + (y_set[j]*x_set[j][0])
                pesos[1] = pesos[1] + (y_set[j]*x_set[j][1])
                if use_bias:
                    bias = bias + y_set[j]
                cant_errores = cant_errores + 1
                error = True
                secuencia.append([pesos[0], pesos[1]])
    print ("Resultado final: ",pesos)    
    if use_bias:
        print("Bias final: ", bias)

    clase_1, clase_2 = clases(x_set, y_set)            
    x, y = zip(*clase_1) #Unzip the x y values
    plt.scatter(x,y, color='blue')

    x, y = zip(*clase_2)
    plt.scatter(x,y, color='red')

    x_line = range(-2, 3)
    y_db = -(pesos[0]/pesos[1]) * x_line - bias/pesos[1] # W1*X1 + W2*X2 + b = 0 -> X2=-(W1*X1 + b)/W2 =   -> Y=-(W1*X1)/W2 - b/W2
    print(y_db)
    plt.xlim(-1,4)
    plt.ylim(-1,4)

    x_inicial = 0
    y_inicial = - (bias/pesos[1])

    plt.grid()
    plt.plot(x_line, y_db)
    plt.quiver(x_inicial, y_inicial, pesos[0], pesos[1])
    plt.show()

x_set = np.array([[2,2],[1, 1],[0.5,2],[1,3]])
y_set = np.array([-1,1,1,-1]) 
perceptron(x_set, y_set)