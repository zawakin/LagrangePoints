# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 02:20:30 2016

@author: Yuki Miyake
"""


import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

m0 = 1.
r0 = np.array([0,0])[:,np.newaxis]

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
    f_v[:,:] = (-m0 * (x - r0) / (np.linalg.norm(x-r0, axis=0))**3  
                + _f)
#    f_v[:,:] = - m0 * (x-r0) / np.linalg.norm(x-r0,axis=0)**3
    f_x[:,:] = v
    return f
    
def RK4(dt, x, f):
    k1 = dt * f(x)
    k2 = dt * f(x+k1/2)
    k3 = dt * f(x+k2/2)
    k4 = dt * f(x+k3)
    return (k1+2*k2+2*k3+k4)/6


def gen():
    global omega
    N = 1
    z = np.empty((4,N))
    x = z[:2,:]
    v = z[2:,:]
#    x[:,:] = np.zeros_like(x) + np.array([1,1])[:,np.newaxis]
#    v[:,:] = np.zeros_like(x)
    x_base = np.array([1,0,])
    x[:,:] = np.zeros((2,N)) + x_base[:,np.newaxis] + np.random.randn(2,N) * 0.1
#    x = np.zeros((2,N))+ np.array([1.,1.])[:,np.newaxis]
    theta = np.arctan2(x[1,:],x[0,:])
    _v = np.sqrt(m0/np.linalg.norm(x,axis=0))
    v[0,:] = - _v * np.sin(theta)
    v[1,:] = _v * np.cos(theta)    
    omega = np.mean(_v / np.linalg.norm(x,axis=0))
    v[:,:] = np.zeros_like(x) + np.array([-0.,0.])[:,np.newaxis]
    dt = 0.001
    t = 0.
    for ti in range(100000):
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
    plt.xlim(-nmax,nmax)
    plt.ylim(-nmax,nmax)
    plt.scatter(z[0],z[1])
#    plt.scatter(L4[0],L4[1], c="r", s=30)
    plt.scatter(r0[0],r0[1], c="b", s=30)
#    plt.scatter(r1[0],r1[1], c="b", s=30)
#    print(z[:2])
    print(omega,np.linalg.norm(z[:2,0]))
    r = np.array([np.cos(omega*t),np.sin(omega*t)])
    plt.scatter(r[0],r[1],c="g", s=40)
#    print("total E = {}".format(((z[2:,0]**2+z[2:,1])**2).sum()))
    
fig = plt.figure()
ani = animation.FuncAnimation(fig, func, gen)
plt.show()