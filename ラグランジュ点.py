# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 22:42:45 2016

@author: Yuki Miyake
"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

m0, m1 = 1, 0.0001
mu = m1 / (m0 + m1)
x_unit = np.array([1.,0.])
r0 = -mu*x_unit
r1 = (1-mu)*x_unit
omega = np.sqrt(m0+m1)
#G_point = 
L4 = (r0 + r1) / 2 + np.array([0.,np.sqrt(3)/2])
L2 = r1 + np.array([0.1,0])
print(L4)
print("mu = {} T = {} ratio = {}".format(m1/(m0+m1)*25.96,2*np.pi/omega,m0/m1))
def f_gravity(z):
    f = np.empty_like(z)
    f_x = f[:2]
    f_v = f[2:]
    x = z[:2]
    v = z[2:]
    inv_v = np.empty_like(v)
    inv_v[0,:] = v[1,:]
    inv_v[1,:] = - v[0,:]
    _f = 2 * omega * inv_v + omega**2 * x
    f_v[:,:] = (-m0 * (x - r0[:,np.newaxis]) / (np.linalg.norm(x-r0[:,np.newaxis], axis=0))**3  
                - m1 * (x - r1[:,np.newaxis]) / (np.linalg.norm(x-r1[:,np.newaxis], axis=0))**3 + _f)
    f_x[:,:] = v
    return f
    
def V_gravity(X,Y):
    R0 = np.sqrt((X-r0[0])**2+(Y-r0[1])**2)
    R1 = np.sqrt((X-r1[0])**2+(Y-r1[1])**2)    
    V = -m0 / R0 - m1 / R1 - 0.5 * omega**2 * R0**2
    return V                
                
def RK4(dt, x, f):
    k1 = dt * f(x)
    k2 = dt * f(x+k1/2)
    k3 = dt * f(x+k2/2)
    k4 = dt * f(x+k3)
    return (k1+2*k2+2*k3+k4)/6


def gen():
    N = 1000
#    x_base = np.array([0.5, 1.])
    x_base = L4
    x = np.zeros((2,N)) + x_base[:,np.newaxis] + np.random.randn(2,N) * 0.00001
#    theta = np.arctan2(x[1,:],x[0,:])
#    _v = np.linalg.norm(x,axis=0) * omega
#    v = np.empty_like(x)
#    v[0,:] = - _v * np.sin(theta)
#    v[1,:] = _v * np.cos(theta)    
#    v[:,:] = np.zeros_like(x) + np.array([-0.,0.])[:,np.newaxis]
    
#    x = np.zeros((2,N)) + x_base[:,np.newaxis]
    v = np.zeros_like(x) #+ np.random.randn(2,N) * 0.01
    z = np.vstack((x,v))
    dt = 0.001
    t = 0.
    for ti in range(10000000):
        t = ti * dt
        z += RK4(dt, z, f_gravity)
        if ti % 100 == 0:

            yield t, z
#            print(t)
            
def func(data):
    t, z = data
    plt.clf()
    plt.title("t = {}".format(t))
    nmax = 2
#    plt.xlim(-nmax,nmax)
#    plt.ylim(-nmax,nmax)
    dx_range = 0.01
    plt.xlim(L4[0]-dx_range,L4[0]+dx_range)
    plt.ylim(L4[1]-dx_range,L4[1]+dx_range)
    plt.scatter(z[0],z[1])
    plt.scatter(L4[0],L4[1], c="r", s=30)
    plt.scatter(r0[0],r0[1], c="b", s=30)
    plt.scatter(r1[0],r1[1], c="b", s=30)
    
    x = np.linspace(-nmax,nmax)
    y = x.copy()
    X,Y = np.meshgrid(x,y)
    V = V_gravity(X,Y)
    plt.contour(X,Y,V,np.linspace(-2,-1.5,20))
    print(V[25])
    print(V.max(),V.min())
#    print("total E = {}".format(((z[2:,0]**2+z[2:,1])**2).sum()))
    
fig = plt.figure()
ani = animation.FuncAnimation(fig, func, gen)
plt.show()