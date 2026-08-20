import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time

from scipy.integrate import simps, trapz


#N_rho x N_z matrix

d_rho = 0.1
d_z = 0.1

L_rho = 10
L_z = 10

N_rho  = int(L_rho/d_rho)
N_z = int(2*L_z/d_z)+1

print ('N_rho:',N_rho)
print ('N_z:',N_z)


rho  = np.linspace(d_rho,L_rho+d_z,N_rho+1)
z  = np.linspace(-L_z-d_z,L_z+d_z+d_z,N_z+3)



delta_rho = rho[1]-rho[0]
delta_z = z[1]-z[0]


rho = rho - 0.5*d_rho

V  = np.zeros((N_rho,N_z))

w  = 1

for i in range(0,N_rho):
    for j in range(0,N_z):
        #V[i,j] = -1/(np.sqrt(rho[i]**2 + z[j+1]**2))
        V[i,j] = -1/np.sqrt(rho[i]**2 + (z[j+1]-1)**2) - 1/np.sqrt(rho[i]**2+(z[j+1]+1)**2)
        #V[i,j] = (1/2)*w*(rho[i+1]**2 + z[j+1]**2)



n = N_rho*N_z

a = 0.5

h_diag    =  a*2*np.ones(n)*(1/delta_rho**2+1/delta_z**2) + V.flatten("F")
#h_off     = -a*np.ones(n-1)/(delta_rho**2)
h_off     = -a*(np.ones(n-1)/(delta_rho**2))#+np.ones(n-1)/(2*delta_rho))
h_off2    = -a*(np.ones(n-1)/(delta_rho**2))
h_off_off = -a*np.ones(n-N_rho)/(delta_z**2)



ii = 1
for i in range(1,n-1):
    if(i%N_rho == 0):
        h_off[i-1] = 0
        h_off2[i-1] = 0
        ii = 1
    else:
        h_off[i-1] *= (rho[ii-1]+delta_rho/2)/np.sqrt(rho[ii-1]*rho[ii])
        ii += 1

h = scipy.sparse.diags([h_diag, h_off, h_off,h_off_off,h_off_off], offsets=[0, -1, 1,-N_rho,N_rho])



t0 = time.time()
epsilon, phi = scipy.sparse.linalg.eigsh(h, k=3, which="SA")
t1 = time.time()
print (np.sort(epsilon).real)

phi0 = phi[:,0].real
phi1 = phi[:,1].real
phi2 = phi[:,2].real

phi0 = np.reshape(phi0, (N_rho, N_z), order='F')
phi1 = np.reshape(phi1, (N_rho, N_z), order='F')
phi2 = np.reshape(phi2, (N_rho, N_z), order='F')


RHO,Z = np.meshgrid(rho[1:N_rho+1],z[1:N_z+1])
"""
plt.figure(1)
plt.pcolor(RHO,Z,abs(phi0)**2)
plt.colorbar()

plt.figure(2)
plt.pcolor(RHO,Z,abs(phi1)**2)
plt.colorbar()

plt.figure(3)
plt.pcolor(RHO,Z,abs(phi2)**2)
plt.colorbar()


plt.show()
"""