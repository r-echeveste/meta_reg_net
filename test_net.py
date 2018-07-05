import numpy as np
import scipy as sp
import sys, os, datetime

from parameters import *
import methods as mt

# Methods to create a connectivity matrix
    
def create_W_mat_standard(s2,out_location):
    W = np.zeros([N,N])
    L = int(np.sqrt(N))
    count = 0
    for i in range(N):
        xi = i%L
        yi = int(i/L)
        for j in range(N):
            if (i != j):
                xj = j%L
                yj = int(j/L)
                d2 = (xi-xj)**2+ (yi-yj)**2
                f = np.exp(-d2/(2*s2))
                if (np.random.rand()<f):
                    W[i,j] = 1.0
                    count += 1
    
    av_n_connections = count /(2.0*N)    
    np.savetxt(out_location+"/W",W)
    print("Average # outgoing connections = ", av_n_connections)
    file = open(out_location+"/n_out_connections.txt","w") 
    file.write("Average # outgoing connections = "+str(av_n_connections)) 
    file.close() 
    # We save an example to visualize at connectivity
    W_ex = np.empty((L,L))
    selected = int(N/2) + int(L/2)
    for neuron in range(N):
        i = neuron%L
        j = int(neuron/L)
        W_ex[i,j] = W[selected,neuron]
        
    np.savetxt(out_location+"/W_example",W_ex)   
    
    return W
    
def create_W_mat_custom(s2,out_location):
    W = np.zeros([N,N])
    L = int(np.sqrt(N))
    count = 0
    for i in range(N):
        xi = i%L
        yi = int(i/L)
        for j in range(N):
            if (i != j):
                xj = j%L
                yj = int(j/L)
                d2 = (xi-xj)**2+ (yi-yj)**2
                f = np.exp(-d2/(2*s2))
                if (np.random.rand()<f):
                    W[i,j] = f
                    count += 1
    W = W * (2.0/sqrt(s2))
    av_n_connections = count /(2.0*N)    
    np.savetxt(out_location+"/W",W)
    print("Average # outgoing connections = ", av_n_connections)
    file = open(out_location+"/n_out_connections.txt","w") 
    file.write("Average # outgoing connections = "+str(av_n_connections)) 
    file.close() 
    # We save an example to visualize at connectivity
    W_ex = np.empty((L,L))
    selected = int(N/2) + int(L/2)
    for neuron in range(N):
        i = neuron%L
        j = int(neuron/L)
        W_ex[i,j] = W[selected,neuron]
        
    np.savetxt(out_location+"/W_example",W_ex)   
    
    return W

############
#   Main   #
############

def main():
    out_location = "results"
    spikes_location = out_location+"/spikes"
    
    load_W_mat = True
    if (not load_W_mat):
        custom_W = True
    
    if not os.path.exists(out_location):
        os.makedirs(out_location)
    if not os.path.exists(spikes_location):
        os.makedirs(spikes_location)
    
    if (load_W_mat):
        print("Loading previous connectivity matrix")
        W = np.loadtxt("W.txt").T
    else:
        if (custom_W):
            s2 = 4.0**2 # connectivity
            W = create_W_mat_custom(s2,out_location)
        else:  
            s2 = 0.9**2 # connectivity
            W = create_W_mat_standard(s2,out_location)
            
    I_app = np.random.normal(0.0, 0.2, N)
        
    sim_time = 100 * 1000 # 100s
    sim_steps = int(1.0*sim_time/dt)
    
    transient_time = 5 * 1000 # 5s
    transient_steps = int(1.0*transient_time/dt)
    
    total_steps = sim_steps + transient_steps
    
    subsamp = 10 
    
    n_points = int(total_steps/subsamp)
    
    steps_1s = int(1000.0/dt)
     
    # Initial conditions
    
    V = np.full((N), -65.1) 
    n = np.full((N), 0.3162)
    x = np.full((N), X)
    y_dock = np.full((N), y_max_dock)
    I_syn_delay = np.zeros((delay_steps,N))
    timelastspike= np.full((N), -50.0)
    Bool_Burst = np.full((N), False)
    Bool_AP = np.full((N), False) 
    Bool_AP_history= np.full((total_steps+keep_bool_AP,N), False)
    Bool_Silent = np.full((N), False)
    state_pool_dock = np.full((N), y_max_dock)
    
    
    V_history = np.empty([n_points,N+1])
    n_history = np.empty([n_points,N+1])
    x_history = np.empty([n_points,N+1])
    y_dock_history = np.empty([n_points,N+1])
    LFP_history = np.empty([n_points,2])
    
    # Simulation
    
    LFP = 0.0
    
    zero_array = np.zeros(N)
    
    print("Start time : ", datetime.datetime.now())
        
    for s in range(total_steps):
        t = (s+1)*dt
        
        # Drawing noise
        if (s > transient_steps/3):
            noise = np.random.normal(0.0, noise_amp, N)
        else:
            noise = zero_array    
    
        # Computing active neurons
        
        Bool_Act = np.logical_and(Bool_AP,np.invert(Bool_Silent))
        
        # Total synaptic current
        
        I_syn_tot = W @ I_syn_delay[0] 
        
        # Integration
        
        (V,n,x,y_dock) = mt.RK_int(V,n,x,y_dock,
                                   Bool_Act,I_app,
                                   I_syn_tot,noise)
        
        # Synaptic current
        
        I_syn_delay[0:delay_steps-1,:] = I_syn_delay[1:delay_steps,:] # Update of the synaptic current.
        
        Bool_AP_prev = np.copy(Bool_AP)
        Bool_AP = (V > V_th)
        
        timelastspike = timelastspike * (np.invert(Bool_AP)) + Bool_AP * t
        F = ((t - timelastspike) < 50 )          # F==0: no spikes in the last 50 ms
        G = np.logical_and(F,np.invert(Bool_Burst))                         
        Bool_Burst[G]= (np.sum(Bool_AP_history[s:s+keep_bool_AP,G],axis = 0) > 10) # no of spikes in the last 200 ms   
        Stop = np.logical_or(np.logical_and(Bool_Burst,(np.invert(F))),(y_dock<y_min_dock))
        Bool_Silent = np.logical_or(Stop,np.logical_and(Bool_Silent,(x > x_RefP)))
        Bool_Burst[Bool_Silent] = False
        state_pool_dock = state_pool_dock * Bool_AP + y_dock * (np.invert(Bool_AP)) # y_dock(t_AP)         
        I_syn_delay[delay_steps-1,:] = K_I * (state_pool_dock - y_dock)* Bool_AP *(np.invert(Bool_Silent))
        Bool_AP_history[s+keep_bool_AP,:] = np.logical_and(Bool_AP,(np.invert(Bool_AP_prev)));    
        
        
        LFP += dt * tau_lfp_inv * (np.average(V)-LFP)
        
        if (s%subsamp == 0):
            s_ = int(s/subsamp)
            V_history[s_,0] = t
            n_history[s_,0] = t
            x_history[s_,0] = t
            y_dock_history[s_,0] = t
            LFP_history[s_,0] = t
            
            V_history[s_,1:] = V
            n_history[s_,1:] = n
            x_history[s_,1:] = x
            y_dock_history[s_,1:] = y_dock
            LFP_history[s_,1] = LFP
            
        if (s%steps_1s==0): print(str(int(s/steps_1s)) + "s of simulation completed")
    
    print("End time : ", datetime.datetime.now())
    
    # Computing spike times and saving results
    print("Saving results...")
    
    index = np.arange(total_steps)
    for neuron in range(N):
        spike_step = np.where(Bool_AP_history[keep_bool_AP:,neuron]==True)
        spike_time = index[spike_step] * dt
        np.savetxt(spikes_location+"/spikes_neuron_"+str(neuron), spike_time)
    
    n_spikes_vs_t = np.stack((index*dt,np.sum(Bool_AP_history[keep_bool_AP:],axis=1)),axis=1)
    
    np.savetxt(out_location+"/n_spikes_vs_t", n_spikes_vs_t)
    
    sample_every = 10 # 10 ms
    
    subsamp_n_spikes = int(sample_every/dt)
    
    n_spikes_vs_t_subsamp = np.empty([int(len(n_spikes_vs_t)/subsamp_n_spikes),2])
    n_spikes_vs_t_subsamp[:,0] = n_spikes_vs_t[::subsamp_n_spikes,0]
    n_spikes_vs_t_subsamp[:,1] = np.sum(n_spikes_vs_t[:,1].reshape(-1, subsamp_n_spikes), axis=1)
    
    np.savetxt(out_location+"/n_spikes_vs_t_"+str(sample_every)+"ms",n_spikes_vs_t_subsamp)
    
    np.savetxt(out_location+"/V_vs_t", V_history)
    np.savetxt(out_location+"/n_vs_t", n_history)
    np.savetxt(out_location+"/x_vs_t", x_history)
    np.savetxt(out_location+"/y_dock_vs_t", y_dock_history)
    np.savetxt(out_location+"/LFP_vs_t", LFP_history)
    

# Call Main
if __name__ == "__main__":
    main()
