import numpy as np
from parameters import *

###############
#   Methods   #
###############

# Methods for the neural model

def alpha_m(V):
    return tau_m_inv * (theta_m -V) / (np.exp(tau_m_inv*(theta_m - V))-1.0)

def beta_m(V):
    return eta_m * np.exp(-sigma_m_inv*(V+65))

def alpha_n(V):
    return 0.1*tau_n_inv * (theta_n -V) / (np.exp(tau_n_inv*(theta_n - V))-1.0)

def beta_n(V):
    return eta_n * np.exp(-sigma_n_inv*(V+65))

def m_infty(V):
    a_m = alpha_m(V)
    b_m = beta_m(V)
    return a_m / (a_m + b_m)

def h(n):
    return (0.89 - 1.1*n)

def I_Na(V,n):
    return g_Na * (m_infty(V)**3) * h(n) * (V - E_Na)

def I_K(V,n):
    return g_K * (n**4) * (V -E_K)

def I_L(V):
    return g_L * (V-E_L)

# Time derivatives
    
def get_derivs (V,n,x,y_dock,Bool_Act,I_app,I_syn_tot):

    # Neural evolution
                
    V_dot = C_inv * (I_app - I_Na(V,n) - I_K(V,n) - I_L(V) + I_syn_tot) # Mem. pot. update
       
    n_dot = alpha_n(V) *(1-n) - beta_n(V) * n      # n update
    
    # Synaptic evolution
    
    x_rel = (x-X)*Bool_Act/X
    x_dot = tau_f_inv * (X-x) + k*(1-x)*Bool_Act         # facilitation var. update
    
    M_factor = y_ref_available * np.exp(-x/m_phi)
          
    y_dock_dot = (   tau_dock_inv * (y_max_dock - y_dock) * (1+x_rel) * M_factor 
                   - tau_rel_inv * y_dock * x_rel   ) # vesicle update
    
    return (V_dot,n_dot,x_dot, y_dock_dot)


# Integrator
def RK_int (V,n,x,y_dock,Bool_Act,I_app,I_syn_tot,noise):
    (V_dot1,n_dot1,x_dot1,y_dock_dot1) = get_derivs (V,n,x,y_dock,
                                        Bool_Act,I_app,I_syn_tot)

    (V_dot2,n_dot2,x_dot2,y_dock_dot2) = get_derivs (V + 0.5 * dt*V_dot1,
                                            n + 0.5 * dt *n_dot1,
                                            x + 0.5 * dt * x_dot1,
                                            y_dock+ 0.5 * dt* y_dock_dot1,
                                            Bool_Act,I_app,I_syn_tot)
    (V_dot3,n_dot3,x_dot3,y_dock_dot3) = get_derivs (V + 0.5 * dt*V_dot2,
                                            n + 0.5 * dt *n_dot2,
                                            x + 0.5 * dt * x_dot2,
                                            y_dock+ 0.5 * dt* y_dock_dot2,
                                            Bool_Act,I_app,I_syn_tot)
    (V_dot4,n_dot4,x_dot4,y_dock_dot4) = get_derivs (V + dt*V_dot3,
                                            n + dt *n_dot3,
                                            x + dt * x_dot3,
                                            y_dock + dt* y_dock_dot3,
                                            Bool_Act,I_app,I_syn_tot)
    
    V_new = V + (dt/6.0) * (V_dot1 + 2.0 * V_dot2 + 2.0 * V_dot3 + V_dot4) + noise
    n_new = n + (dt/6.0) * (n_dot1 + 2.0 * n_dot2 + 2.0 * n_dot3 + n_dot4)
    x_new = x + (dt/6.0) * (x_dot1 + 2.0 * x_dot2 + 2.0 * x_dot3 + x_dot4)
    y_dock_new = y_dock + (dt/6.0)*(y_dock_dot1+2.0*y_dock_dot2+2.0*y_dock_dot3+y_dock_dot4)
    
    return (V_new,n_new,x_new,y_dock_new)
  
