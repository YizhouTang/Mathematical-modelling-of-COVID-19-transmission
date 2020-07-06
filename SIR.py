import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SIR():

    def __init__(self,y,t,N,beta,gamma):
        self.y = y
        self.t = t
        self.N = N
        self.beta = beta
        self.gamma = gamma


    def intSIR(self):
        return odeint(deriv,self.y, self.t, args=(self.N, self.beta, self.gamma)).T

    def plot_normalized_curves(self):
        S,I,R = self.intSIR()

        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(self.t, S / self.N, 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(self.t, I / self.N, 'r', alpha=0.5, lw=2, label='Infected')
        ax.plot(self.t, R / self.N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Normalized')
        ax.set_ylim(0, 1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()

    def plot_individual_curves(self):
        S,I,R = self.intSIR()
        plt.figure()
        plt.plot(self.t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
        plt.xlabel("dates")
        plt.ylabel("count")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.t, I, 'r', alpha=0.5, lw=2, label='Infected')
        plt.xlabel("dates")
        plt.ylabel("count")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(self.t,R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
        plt.xlabel("dates")
        plt.ylabel("count")
        plt.legend()
        plt.show()

def deriv(y,t,N,beta,gamma):
    S,I,R = y
    dSdt = - beta * S * I/N
    dIdt = beta * S * I/N - gamma * I
    dRdt = gamma * I
    return dSdt,dIdt,dRdt

