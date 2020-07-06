from SIR import SIR
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.special import lambertw
import matplotlib.pyplot as plt
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


optimal = minimize(loss, [0.001, 0.001], args=(data, confirmed,recovered, S0, I0, R0), method='L-BFGS-B', bounds=[(0.00000001, 1), (0.00000001, 1)])
beta, gamma = optimal.x
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
