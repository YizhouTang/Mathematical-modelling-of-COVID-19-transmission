import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.special import lambertw
from matplotlib import pyplot as plt

from scipy.integrate import odeint

# Define parameters
# t_max = 100
# dt = .1
# t = np.linspace(0, t_max, int(t_max / dt) + 1)
# N = 37590000
# init_vals = 1 - 1 / N, 1 / N, 0, 0


# alpha = 0.04
# beta = 0.4
# gamma = 0.2

class SEIR():

    def __init__(self,y,t,N,alpha,beta,gamma,rho):
        self.y = y
        self.t = t
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho

    def base_seir_model(self,init_vals, params, t):
        S_0, E_0, I_0, R_0 = init_vals
        S, E, I, R = [S_0], [E_0], [I_0], [R_0]
        alpha, beta, gamma = params
        dt = t[1] - t[0]
        for _ in t[1:]:
            next_S = S[-1] - (beta*S[-1]*I[-1])*dt
            next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt
            next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
            next_R = R[-1] + (gamma*I[-1])*dt
            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
        return np.stack([S, E, I, R]).T

    def seir_model_with_soc_dist(self,init_vals, params, t):
        S_0, E_0, I_0, R_0 = init_vals
        S, E, I, R = [S_0], [E_0], [I_0], [R_0]
        alpha, beta, gamma, rho = params
        dt = t[1] - t[0]
        for _ in t[1:]:
            next_S = S[-1] - (rho*beta*S[-1]*I[-1]) * dt
            next_E = E[-1] + (rho*beta*S[-1]*I[-1] - alpha*E[-1])*dt
            next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
            next_R = R[-1] + (gamma*I[-1])*dt

            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
        return np.stack([S, E, I, R]).T

    def plt_base_seir(self,init_vals, params, t):
        #params = alpha, beta, gamma
        # Run simulation
        results = self.base_seir_model(init_vals, params, t)

        S = results[:,0]
        E = results[:,1]
        I = results[:,2]
        R = results[:,3]

        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(t, S , 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(t, E , 'y', alpha=0.5, lw=2, label='Exposed')
        ax.plot(t, I , 'r', alpha=0.5, lw=2, label='Infected')
        ax.plot(t, R , 'g', alpha=0.5, lw=2, label='Recovered')
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
        plt.title("SEIR")
        plt.show()

    def plt_seir_model_with_soc_dist(self,alpha,beta,gamma,rho,init_vals, t):
        params = alpha, beta, gamma,rho
        # Run simulation
        sd_results = self.seir_model_with_soc_dist(init_vals, params, t)

        S = sd_results[:,0]
        E = sd_results[:,1]
        I = sd_results[:,2]
        R = sd_results[:,3]

        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(t, S/self.N , 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(t, E/self.N  , 'y', alpha=0.5, lw=2, label='Exposed')
        ax.plot(t, I/self.N  , 'r', alpha=0.5, lw=2, label='Infected')
        ax.plot(t, R/self.N  , 'g', alpha=0.5, lw=2, label='Recovered')
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
        plt.title("SEIR with social distancing - Rho = "+str(rho))
        plt.show()

    def plt_soc_dist_different_rhos(self,alpha,beta,gamma,init_vals,t):

        rho_list = [0, 0.25,0.5]#[0, 0.33, 0.66, 1]
        legends = ['0', '0.25','0.5']#['0', '0.33', '0.66', '1']

        # Susceptible
        plt.figure()
        for rho in rho_list:
            params = alpha, beta, gamma, rho
            sd_results = self.seir_model_with_soc_dist(init_vals, params, t)
            S = sd_results[:, 0]
            plt.plot(S)
        plt.title("Effects of social distancing - Susceptible")
        plt.legend(["Rho = " + x for x in legends])
        plt.xlabel("Days")
        plt.ylabel("Count")
        plt.savefig('Effects of social distancing - Susceptible.png')
        plt.show()

        # Exposed
        plt.figure()
        for rho in rho_list:
            params = alpha, beta, gamma, rho
            sd_results = self.seir_model_with_soc_dist(init_vals, params, t)
            E = sd_results[:, 1]
            plt.plot(E)
        plt.xlabel("Days")
        plt.ylabel("Count")
        plt.title("Effects of social distancing - Exposed")
        plt.legend(["Rho = " + x for x in legends])
        plt.savefig('Effects of social distancing - Exposed.png')
        plt.show()

        # Infected
        plt.figure()
        for rho in rho_list:
            params = alpha, beta, gamma, rho
            sd_results = self.seir_model_with_soc_dist(init_vals, params, t)
            I = sd_results[:, 2]
            plt.plot(I)
        plt.xlabel("Days")
        plt.ylabel("Count")
        plt.title("Effects of social distancing - Infected")
        plt.legend(["Rho = " + x for x in legends])
        plt.savefig('Effects of social distancing - Infected.png')
        plt.show()

        # Recovered
        plt.figure()
        for rho in rho_list:
            params = alpha, beta, gamma, rho
            sd_results = self.seir_model_with_soc_dist(init_vals, params, t)
            R = sd_results[:, 3]
            plt.plot(R)
        plt.xlabel("Days")
        plt.ylabel("Count")
        plt.title("Effects of social distancing - Recovered")
        plt.legend(["Rho = " + x for x in legends])
        plt.savefig('Effects of social distancing - Recovered.png')
        plt.show()

#Rho = 0 indicates everyone is locked down and quarantined while 1 is equivalent to our base case above.

#rho = 0.5
#rho_list = [0,0.33,0.66,1]
#legends = ['0','0.2','0.4','0.6','0.8','1']

#for rho in rho_list:

# https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(20)30074-7/fulltext
#
#
# params = alpha, beta, gamma, rho
#
#
# #Rho from 0 to 1

#
# def MSE(C,T,y_true):
#     alpha = C[0]
#     beta = C[1]
#     gamma = C[2]
#     rho = C[3]
#     params = alpha, beta, gamma, rho
#     y_hat = seir_model_with_soc_dist(init_vals, params, T)
#
#     return np.mean((y_hat - y_true)**2)


#
# T = list(range(1, len(confirmed) + 1))
# y_true = np.array(confirmed)
#
# initial_guess = [0.1,0.1]
#
# res = minimize(MSE,initial_guess ,method = 'Nelder-Mead', args=(T,y_true))
#
#
#
# plt.figure()
# for rho in rho_list:
#     params = alpha, beta, gamma, rho
#     sd_results = seir_model_with_soc_dist(init_vals, params, t)
#     S = sd_results[:,0]
#     E = sd_results[:,1]
#     I = sd_results[:,2]
#     R = sd_results[:,3]
#     plt.plot(S)
#     plt.plot(E)
#     plt.plot(I)
#     plt.plot(R)
# plt.title("Effects of social distancing - Recovered")
# plt.legend( ["Rho = "+x for x in legends])
# plt.show()
#
