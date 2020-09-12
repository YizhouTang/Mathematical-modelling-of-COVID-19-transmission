from SIR import SIR
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.special import lambertw
import matplotlib.pyplot as plt
from SEIR import SEIR
from scipy.integrate import odeint

#### Models
def SIR(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def SEIR(y, t, N, alpha,beta, gamma,rho):
    S, E,I, R = y
    dSdt = -beta * S * E - rho * beta * S * I
    dEdt = rho * beta * S * I + beta * S * E - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt,dIdt, dRdt

#Compute Mean-Squared Error for SIR
def SIR_MSE( point, susceptible, infected, recovered, s_0, i_0, r_0):
    timeSteps = np.linspace(0, len(infected), len(infected))
    beta, gamma = point

    # Initial conditions vector
    y0 = s_0, i_0, r_0
    # Integrate the SIR equations over the time grid, t.
    solution = odeint(SIR, y0, timeSteps, args=(N, beta, gamma))
    S, I, R = solution.T

    S_loss = np.sqrt(np.mean((S - susceptible) ** 2))
    I_loss = np.sqrt(np.mean((I - infected) ** 2))
    R_loss = np.sqrt(np.mean((R - recovered) ** 2))
    return (S_loss + I_loss + R_loss)/3

#Compute Mean-Squared Error for SEIR (with social distancing)
def SEIR_MSE(point, susceptible,infected, recovered, s_0, e_0,i_0, r_0):
    timeSteps = np.linspace(0, len(infected), len(infected))
    alpha, beta, gamma, rho = point

    # Initial conditions vector
    y0 = s_0, e_0, i_0, r_0
    # Integrate the SEIR equations over the time grid, t.
    solution = odeint(SEIR, y0, timeSteps, args=(N, alpha,beta, gamma,rho))
    S,E, I, R = solution.T

    S_loss = np.sqrt(np.mean((S - susceptible) ** 2))
    I_loss = np.sqrt(np.mean((I - infected) ** 2))
    R_loss = np.sqrt(np.mean((R - recovered) ** 2))
    return (S_loss+I_loss+R_loss)/3

#### Plotting functions

def plot_individual_curves(S, I, R, t):
    plt.figure()
    plt.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    plt.xlabel("Days")
    plt.ylabel("Count")
    plt.legend()
    plt.title("SIR Model - Susceptible")
    plt.show()
    plt.savefig("SIR Model - Susceptible.png")

    plt.figure()
    plt.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    plt.xlabel("Days")
    plt.ylabel("Count")
    plt.legend()
    plt.title("SIR Model - Infected")
    plt.show()
    plt.savefig("SIR Model - Infected.png")

    plt.figure()
    plt.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered/Removed')
    plt.xlabel("Days")
    plt.ylabel("Count")
    plt.legend()
    plt.title("SIR Model - Recovered")
    plt.show()
    plt.savefig("SIR Model - Recovered.png")


#Load data
df = pd.read_csv('data.csv')#,parse_dates = True,index_col='date')
df.ffill(inplace = True)
df.dropna(inplace = True)
# Total population, N.
N = 37590000
infected = df['numconf'].values
recovered = df['numrecover'].values + df['numdeaths'].values
susceptible = N - infected- recovered

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = infected[0], recovered[0]
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# A grid of time points (in days)
DAYS = 1000#len(infected)
t = np.linspace(0, DAYS, DAYS)
SHOW_PLOTS = True


# Initial conditions vector
y0 = S0, I0, R0

#Curve fit SIR model to canadian data
optimal = minimize(SIR_MSE,
                   [0.0001, 0.0001],
                   args=(susceptible, infected,recovered, S0, I0, R0),
                   bounds=[(0, 1), (0, 1)],
                   method = 'L-BFGS-B'
                   )
beta, gamma = optimal.x
print("---------------")
print("SIR Model - Optimal Parameters:")
print("Beta: ",beta)
print("Gamma: ",gamma)


#####LambertW section
ratio = gamma / beta
ratio = [x / 100 for x in range(1, 200)]

uf = []
for i in ratio:
    sol = (lambertw(-999 / 1000 * i * np.exp(-i)) / i + 1)
    uf.append(np.real(sol))

plt.figure()
plt.plot(ratio, uf)
plt.title("Phase Transition at gamma/beta = 1")
plt.xlabel("gamma/beta")
plt.show()


# Initialize model based on optimal estimated parameters
# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
solution = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R = solution.T



plot_individual_curves(S, I, R, t)
E0 = 0

optimal = minimize(SEIR_MSE,
                   [0.0001, 0.0001,0.0001,0.0001],
                   args=(susceptible,infected,recovered, S0, E0,I0, R0),
                   bounds=[(0, 1), (0, 1),(0, 1),(0, 1)],
                   method = 'L-BFGS-B'
                   )
alpha, beta, gamma, rho = optimal.x

print("---------------")
print("SEIR Model with Social Distancing - Optimal Parameters:")
print("Alpha: ",alpha)
print("Beta: ",beta)
print("Gamma: ",gamma)
print("Rho: ",rho)


# Initialize model based on optimal estimated parameters
params = alpha, beta, gamma, rho

# Initial conditions vector
y0 = S0, E0,I0, R0
# Integrate the SIR equations over the time grid, t.
solution = odeint(SEIR, y0, t, args=(N, alpha,beta, gamma,rho))
S, E,I, R = solution.T




#
# #SEIR plot
# SEIR_mdl.plt_seir_model_with_soc_dist(alpha, beta, gamma, rho, init_vals, t)

#SEIR_mdl.plt_soc_dist_different_rhos(alpha,beta,gamma,init_vals,t)
rho_list = [0, 0.33, 0.66, 1]  # [0, 0.25, 0.5]  #
legends = ['0', '0.33', '0.66', '1']  # ['0', '0.25', '0.5']  #

# Susceptible
plt.figure()
for rho_i in rho_list:
    params = alpha, beta, gamma, rho_i
    solution = odeint(SEIR, y0, t, args=(N, alpha, beta, gamma, rho))
    S, E, I, R = solution.T
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
    solution = odeint(SEIR, y0, t, args=(N, alpha, beta, gamma, rho))
    S, E, I, R = solution.T
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
    solution = odeint(SEIR, y0, t, args=(N, alpha, beta, gamma, rho))
    S, E, I, R = solution.T
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
    solution = odeint(SEIR, y0, t, args=(N, alpha, beta, gamma, rho))
    S, E, I, R = solution.T
    plt.plot(R)
plt.xlabel("Days")
plt.ylabel("Count")
plt.title("Effects of social distancing - Recovered")
plt.legend(["Rho = " + x for x in legends])
plt.savefig('Effects of social distancing - Recovered.png')
plt.show()
