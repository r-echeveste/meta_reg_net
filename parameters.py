import numpy as np

##################
#   Parameters   #
##################

#--------------------#
# Network parameters #
#--------------------#

N = 400

#-------------------#
# Neural parameters #
#-------------------#


# Conductances
g_Na = 120.0
g_K = 36.0
g_L = 0.3

# Capacitance
 
C = 1.0
C_inv = 1.0/C

# Action Potential threshold

V_th = -58.0

# Equilibrium potentials

E_Na = 50.0
E_K = -77.0
E_L = -54.4

# Time params

dt = 0.05                      # Integration time-step 0.05 ms
dt_inv = 1.0/dt

delay_time = 1.0                     # Delay for chemical synapses: 1 ms.
delay_steps = int(delay_time/dt)

keep_bool_AP = int(200.0/dt)     # Number of steps kept in memory 
                                 # to determine the beginning of 
                                 # the refractory period. (200 ms).

# Voltage params

noise_sd = 0.89

noise_amp = C_inv * np.sqrt(dt) * noise_sd


# Vesicle params

y_max_dock = 0.18
y_min_dock = 0.04
y_ref_available = 0.32
X = 0.3

tau_f = 700.0
tau_dock = 738.0
tau_rel = 47.0

tau_f_inv = 1.0/tau_f
tau_dock_inv = 1.0/tau_dock
tau_rel_inv = 1.0/tau_rel

k = 0.08

# Synaptic current params

K_I = 2666 #  KI converts the proportion of fused vesicles during an AP into a postsynaptic current
x_RefP = 0.31 # x for refractory period

# m and n params

tau_m = 10.0
tau_m_inv = 1.0/tau_m
theta_m = -40.0
eta_m = 4.0
sigma_m = 18.0
sigma_m_inv = 1.0/sigma_m

tau_n = 10.0
tau_n_inv = 1.0/tau_n
theta_n = -55.0
eta_n = 0.125
sigma_n = 80.0
sigma_n_inv = 1.0/sigma_n

# LFP 
tau_lfp = 1.0
tau_lfp_inv = 1.0/tau_lfp

# Metabolic target

m_phi = 0.5

# Facilitation variable

alpha_facil = 0.0

