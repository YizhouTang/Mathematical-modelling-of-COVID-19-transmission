import numpy as np

from matplotlib import pyplot as plt

def base_seir_model(init_vals, params, t):
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

# Define parameters
t_max = 100
dt = .1
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
init_vals = 1 - 1/N, 1/N, 0, 0
alpha = 0.2
beta = 1.75
gamma = 0.5
params = alpha, beta, gamma
# Run simulation
results = base_seir_model(init_vals, params, t)


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


def seir_model_with_soc_dist(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (rho*beta*S[-1]*I[-1] + beta * S[-1] * E[-1])*dt # added -beta*S[-1]*E[-1]
        next_E = E[-1] + (rho*beta*S[-1]*I[-1] - alpha*E[-1] + beta*S[-1]*E[-1])*dt #added + beta*S[-1]*E[-1]
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T

#Rho = 0 indicates everyone is locked down and quarantined while 1 is equivalent to our base case above.

rho = 0

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


#Rho from 0 to 1
rho_list = [0,0.2,0.4,0.6,0.8,1]
legends = ['0','0.2','0.4','0.6','0.8','1']
plt.figure()
for rho in rho_list:
    params = alpha, beta, gamma, rho
    sd_results = seir_model_with_soc_dist(init_vals, params, t)
    E = sd_results[:,1]
    plt.plot(E)

plt.legend(legends)
plt.show()
