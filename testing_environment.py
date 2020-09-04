from SIR import SIR
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.special import lambertw
import matplotlib.pyplot as plt
from SEIR import seir_model_with_soc_dist
df = pd.read_csv('data.csv')#,parse_dates = True,index_col='date')
df.ffill(inplace = True)
confirmed = df['numconf'].values
recovered = df['numrecover'].values + df['numdeaths'].values

plt.figure()
plt.plot(confirmed)
plt.plot(recovered)
plt.legend(['Infected','Recovered'])
plt.title("Canada COVID-10 Stats")
plt.show()


# Total population, N.
N = 37590000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = confirmed[0], recovered[0]
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0


data = N - confirmed- recovered

def loss(point, data,infected, recovered, s_0, i_0, r_0):
    size = len(infected)
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I/N, beta*S*I/N-gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l0 = np.sqrt(np.mean((solution.y[0] - data)**2))
    l1 = np.sqrt(np.mean((solution.y[1] - infected)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    #print(l0,l1,l2)
    #alpha = 0.5
    return l0+l1+l2

#optimal = minimize(loss, [0.001, 0.001], args=(data, confirmed,recovered, S0, I0, R0), method='L-BFGS-B', bounds=[(0.00000001, 1), (0.00000001, 1)])
#beta, gamma = optimal.x

#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7277829/#B11-ijerph-17-03535
# Peng et al. (2020)
# values for for Wuhan

beta = 1
gamma = 0.5
print("Beta: ",beta)
print("Gamma: ",gamma)

# A grid of time points (in days)
t = np.linspace(0, 365, 365)

# Initial conditions vector
y0 = S0, I0, R0
# Initialize model
mdl = SIR(y0,t,N,beta,gamma)
#Plot
mdl.plot_normalized_curves()
mdl.plot_individual_curves()


#####LambertW section
ratio = gamma/beta
plt.plot()
ratio = [x/100 for x in range(1,200)]

uf = []
for i in ratio:
    sol = (lambertw(-999 / 1000 * i * np.exp(-i)) / i + 1)
    uf.append(np.real(sol))
plt.figure()
plt.plot(ratio,uf)
plt.title("Phase Transition at gamma/beta = 1")
plt.xlabel("gamma/beta")
plt.show()

def SEIR_loss(point, data,infected, recovered, s_0, e_0,i_0, r_0):
    size = len(infected)
    alpha, beta, gamma, rho = point

    def SEIR(t, y):
        S = y[0]
        E = y[1]
        I = y[2]
        R = y[3]

        return [-rho * beta * S * I / N,
                rho * beta * S * I / N - gamma * I,
                (alpha * E - gamma * I),
                (gamma * I)
                ]
        # print(l0,l1,l2)

    solution = solve_ivp(SEIR, [0, size], [S0, E0, I0, R0], t_eval=np.arange(0, size, 1), vectorized=True)
    l0 = np.sqrt(np.mean((solution.y[0] - data) ** 2))
    l1 = np.sqrt(np.mean((solution.y[2] - confirmed) ** 2))
    l2 = np.sqrt(np.mean((solution.y[3] - recovered) ** 2))
    #print(l0,l1,l2)
    #alpha = 0.5
    return l0+l1+l2
E0 = 0
#optimal = minimize(SEIR_loss, [0.001, 0.001,0.001,0.001], args=(data, confirmed,recovered, S0, E0,I0, R0), method='L-BFGS-B', bounds=[(0.00000001, 1), (0.00000001, 1),(0.00000001, 1),(0.00000001, 1)])
#alpha, beta, gamma, rho = optimal.x

alpha = 0.085
beta = 1
gamma = 0.5
rho = 0.5
print("Alpha: ",alpha)
print("Beta: ",beta)
print("Gamma: ",gamma)
#print("Rho: ",rho)


# Define parameters
t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 37590000
init_vals = 1 - 1/N,1/N, 0, 0
params = alpha, beta, gamma, rho

sd_results = seir_model_with_soc_dist(init_vals, params, t)

S = sd_results[:,0]
E = sd_results[:,1]
I = sd_results[:,2]
R = sd_results[:,3]

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
plt.title("SEIR with social distancing")
plt.show()




