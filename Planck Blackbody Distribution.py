import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize



df = pd.read_csv('data.csv')#,parse_dates = True,index_col='date')
df.ffill(inplace = True)
confirmed = df['numconf'].values
recovered = df['numrecover'].values + df['numdeaths'].values
#
# plt.figure()
# plt.plot(confirmed)
# plt.legend(['Infected'])
# plt.title("Canada COVID-10 Stats")
# plt.show()


def BlackBodyDistribution(C1,C2,T):
    # numerator exp(C*x)-1

    # (C2*T^3) / exp(C1*T) -1
    # Fit the above equation so that the predicted values are as close to I as possible

    # more detailed reference for reference #12
    # Kermack, W. O. and McKendrick, A. G. "A Contribution to the Mathematical Theory of Epidemics." Proc. Roy. Soc. Lond. A 115, 700-721, 1927

    # add the second Nikolauou paper reference

    # Kroger & schlickeiser gaussian distribution

    y_hat = []
    for t in T:
        y_hat.append((C2*t**2) / (np.exp(C1*t)- 1) )
    return np.array(y_hat)

def MSE(C,T,y_true):
    C1 = C[0]
    C2 = C[1]
    y_hat = BlackBodyDistribution(C1,C2,T)

    return np.mean((y_hat - y_true)**2)



T = list(range(1, len(confirmed) + 1))
y_true = np.array(confirmed)
initial_guess = [0.1,0.1]
res = minimize(MSE,initial_guess ,method = 'Nelder-Mead', args=(T,y_true))

print(res)
C = res.x

T = list(range(1,1000))
y_pred = BlackBodyDistribution(C[0],C[1],T)
plt.plot(T,y_pred)
#plt.plot(T,y_true)
plt.legend(['y_pred'])#,'y_true'])
plt.title("Infection Modeling - Planck Blackbody Distribution")
plt.savefig('Infection Modeling - Planck Blackbody Distribution (T^2).png')
plt.show()
#
# # Total population, N.
# N = 37590000
# # Initial number of infected and recovered individuals, I0 and R0.
# I0, R0 = confirmed[0], recovered[0]
# # Everyone else, S0, is susceptible to infection initially.
# S0 = N - I0 - R0
