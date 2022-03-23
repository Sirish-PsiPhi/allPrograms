import numpy as np
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show

def radial_kernel(x0,X,tau):
    return np.exp(np.sum((X-x0) ** 2,axis=1) / (-2 * tau * tau))

def local_regression(x0,X,Y,tau):
    x0 = np.r_[1,x0]
    X = np.c_[np.ones(len(X)) , X]

    xtw = X.T * radial_kernel(x0,X,tau)

    beta = np.linalg.pinv(xtw @ X) @ xtw @ Y

    return x0 @ beta

X = np.linspace(-3,3,num=1000)
print("The Data Set ( 10 Samples) X :\n", X[1:10])
Y = np.log(np.abs(X ** 2 - 1) + .5)
print("The Fitting Curve Data Set (10 Samples) Y :\n", Y[1:10])

X += np.random.normal(scale=.1,size=1000)
print("Normalised (10 Samples) X :\n", X[1:10])

domain = np.linspace(-3,3,num=1000)
print(" Xo Domain Space(10 Samples) :\n", domain[1:10])

def plot_lwr(tau):
    prediction = [local_regression(x0,X,Y,tau) for x0 in domain]
    plot = figure(plot_width=400,plot_height=400)
    plot.title.text = f"tau={tau}"
    plot.scatter(X,Y,alpha=.3)
    plot.line(domain,prediction,line_width=2,color='red')
    return plot

show(gridplot([
    [plot_lwr(10.),plot_lwr(1.)],
    [plot_lwr(0.1),plot_lwr(0.01)]
]))