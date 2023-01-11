import numpy as np
import matplotlib.pyplot as plt
def estimate_coef(x,y):
    n=np.size(x)
    m_x,m_y=np.mean(x),np.mean(y)
    SS_xy=np.sum(y*x)-n*m_y*m_x
    SS_xx=np.sum(x*x)-n*m_x*m_x
    b_1=SS_xy/SS_xx
    b_0=m_y-b_1*m_x
    return(b_0,b_1)

def plot_regression_line(x,y,b_0,b_1):
    plt.scatter(x,y,color='m',marker='o',s=30)
    y_pred=b_0+b_1*x
    plt.plot(x,y_pred,color='g')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()
    
def main():
    x=np.array([0,1,2,3,4,5,6,7,8,9])
    y=np.array([1,3,2,5,7,8,8,9,10,12])
    b_0,b_1=estimate_coef(x,y)
    print('Estimated coefficients:\nb_0={}\nb_1={}'.format(b_0,b_1))
    plot_regression_line(x,y,b_0,b_1)
main()