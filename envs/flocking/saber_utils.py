import numpy as np
import math

A, B = 5, 5  # 0<A<=B 
C = np.abs(A-B)/np.sqrt(4*A*B) # phi
H = 0.2 # Bump function




EPSILON = 0.1

def bump_function(z,H=0.2):
    Ph = np.zeros_like(z)
    # Ph[z>=H] = (1 + np.cos(np.pi*(z[z<=1]-H)/(1-H)))/2
    Ph[z>=H] = (1 + np.cos(np.pi*(z[z>=H]-H)/(1-H)))/2
    Ph[z<H] = 1
    Ph[z<0] = 0
    return Ph

def phi(z):
    phi_val = ((A + B) * sigma_1(z+C) + (A-B)) / 2
    return phi_val

def phi_alpha(z, range, distance, H=0.2):
    r_alpha = sigma_norm([range])
    d_alpha = sigma_norm([distance])
    phi_alpha_val = bump_function(z/r_alpha, H) * phi(z-d_alpha)
    return phi_alpha_val

def sigma_1(z, base=1.0):
    sigma_1_val = z/np.sqrt(base + z**2)
    return sigma_1_val

def sigma_norm(z):
    sigma_norm_val = (np.sqrt(1+EPSILON*np.linalg.norm(z, axis=-1, keepdims=True)**2)-1) / EPSILON 
    return sigma_norm_val

def sigma_grad(z):
    sigma_grad_val = z/np.sqrt(1 + EPSILON*np.linalg.norm(z, axis=-1, keepdims=True)**2)
    return sigma_grad_val

def get_adjacency_matrix(nodes, r):
    return np.array([np.linalg.norm(nodes[i,:2]-nodes[:,:2], axis=-1)<=r for i in range(len(nodes))])

def get_adjacency_matrix_obs(nodes,obs,r):
    return np.array([np.linalg.norm(obs[:,:2]-nodes[:,:2], axis=-1)<=r for i in range(len(nodes))])

def get_a_ij(q_i, q_js, range, H=0.2):
    r_alpha = sigma_norm([range])
    a_ij = bump_function(sigma_norm(q_js-q_i)/r_alpha, H)
    return a_ij

def get_n_ij(q_i, q_js):
    n_ij = sigma_grad(q_js-q_i)
    return n_ij

def get_adj_min_dist(nodes):
    dist=np.array([np.linalg.norm(nodes[i,:2]-nodes[:,:2], axis=-1) for i in range(len(nodes))])
    np.fill_diagonal(dist,np.inf)
    return np.min(dist)

def obs_projection_sphere(state, RK,yk):
    # RK: radius of obstacle
    # YK: center location of obstacle
    number=state.shape[0]
    NUMBER_OF_OBS=yk.shape[0]

    diff=state[:,:2].reshape(number,1,2)-yk.reshape(1,NUMBER_OF_OBS,2)
    mu=RK/np.linalg.norm(diff,axis=2)
    ak=(diff)/np.linalg.norm(diff,axis=2).reshape(number,NUMBER_OF_OBS,1)
    ak=ak.transpose(1,0,2)
    ak=ak.reshape(ak.shape[0],ak.shape[1],1,ak.shape[2])
    P=np.array([np.identity(number) for i in range(NUMBER_OF_OBS)])
    P=P-np.einsum("bijk,bikj->bij",ak,ak.transpose(0,1,3,2))
    if number >1:
        P=np.divide(P,np.linalg.det(P).reshape(NUMBER_OF_OBS,1,1))

    q_hat=(mu.reshape(number,NUMBER_OF_OBS,1)*state[:,:2].reshape(number,1,2))+(1-mu.reshape(number,NUMBER_OF_OBS,1))*yk.reshape(1,NUMBER_OF_OBS,2) # position 3,2,2??
    q_hat=q_hat.transpose(1,0,2)
    p_hat=(P@state[:,2:])*(mu.T).reshape(NUMBER_OF_OBS,number,1) #2,3,2
    return np.dstack([q_hat,p_hat]) # number_obs, number, 4

def xy(r,phi,yk):
    return r*np.cos(phi)+yk[0], r*np.sin(phi)+yk[1]