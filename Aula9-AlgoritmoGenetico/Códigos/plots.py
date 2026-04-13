import numpy as np
import matplotlib.pyplot as plt

PI = 3.141592684
plt.rcParams.update({'font.size': 18})

def quad(x):
    return x*x
def rast(x):
    return 10+x*x-10*np.cos(2*PI*x)

func = rast

xdata = []
for i in range(4):
    str_data = input("Dados de entrada: ")
    xdata.append(np.loadtxt(str_data)[:2000])

x_range = np.arange(-5,5,0.001)
plt.plot(x_range,func(x_range))
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.show()

eta = []

for i in range(4):
    eta.append(eval(input("Valor de eta: ")))
    plt.plot(xdata[i],label=r"$\eta = "+str(eta[i])+"$")
plt.legend()
plt.xlabel("Número de iterações")
plt.ylabel(r"$x$")
plt.show()

for i in range(4):
    plt.plot(func(xdata[i]),label=r"$\eta = "+str(eta[i])+"$")
plt.legend()
plt.xlabel("Número de iterações")
plt.ylabel(r"$f(x)$")
plt.yscale("log")
plt.show()