# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 00:42:47 2016

@author: Yuki Miyake
"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

mu = 0.01
x_unit = np.array([1.,0.])
r0 = -mu*x_unit
r1 = (1-mu)*x_unit
L4 = (r0 + r1) / 2 + np.array([0.,np.sqrt(3)/2])

C = np.sort(np.array([3+3**(4/3)*mu**(2/3) - 10*mu/3,
              3+3**(4/3)*mu**(2/3) - 14*mu/3,3+mu,3-mu]))
              
print("criteria = {}".format(mu*25.96))
def f_gravity(z):
    f = np.empty_like(z)
    f_x = f[:2]
    f_v = f[2:]
    x = z[:2]
    v = z[2:]
#    inv_v = np.empty_like(v)
#    inv_v[0,:] = v[1,:]
#    inv_v[1,:] = - v[0,:]
#    _f = 2  * inv_v + x
#    f_v[:,:] = (-(1-mu)*(x - r0[:,np.newaxis]) / (np.linalg.norm(x-r0[:,np.newaxis], axis=0))**3  
#                - mu*(x - r1[:,np.newaxis]) / (np.linalg.norm(x-r1[:,np.newaxis], axis=0))**3 + _f)
    rho0 = np.linalg.norm(x-r0[:,np.newaxis],axis=0)
    rho1 = np.linalg.norm(x-r1[:,np.newaxis],axis=0)
    f_vx = 2 * v[1] - (1-mu) * (x[0]-r0[0]) / rho0**3 - mu * (x[0]-r1[0]) / rho1**3 + x[0]
    f_vy = -2 * v[0] - (1-mu) * (x[1]-r0[1]) / rho0**3 - mu * (x[1]-r1[1]) / rho1**3 + x[1]
    f_v[0] = f_vx
    f_v[1] = f_vy
    f_x[:,:] = v
    return f
    
def V_gravity(X,Y):
    R0 = np.sqrt((X-r0[0])**2+(Y-r0[1])**2)
    R1 = np.sqrt((X-r1[0])**2+(Y-r1[1])**2)    
    V = 2*(1-mu) / R0 + 2 * mu / R1 + (X*X+Y*Y)
    return V                
                
def RK4(dt, x, f):
    k1 = dt * f(x)
    k2 = dt * f(x+k1/2)
    k3 = dt * f(x+k2/2)
    k4 = dt * f(x+k3)
    return (k1+2*k2+2*k3+k4)/6

def gen():
    xs = np.array([])
    N =1
    x_base = L4
    x = np.zeros((2,N)) + x_base[:,np.newaxis] + np.random.randn(2,N) * 0.01
#    x = np.zeros((2,N)) + 2 * np.random.rand(2,N) - 1
#    theta = np.arctan2(x[1,:],x[0,:])
#    _v = np.linalg.norm(x,axis=0) * omega
#    v = np.empty_like(x)
#    v[0,:] = - _v * np.sin(theta)
#    v[1,:] = _v * np.cos(theta)    
#    v[:,:] = np.zeros_like(x) + np.array([-0.,0.])[:,np.newaxis]
    
#    x = np.zeros((2,N)) + x_base[:,np.newaxis]+ np.random.randn(2,N) * 0.01
    v = np.zeros_like(x) #+ np.random.randn(2,N) * 0.01
    z = np.vstack((x,v))
    dt = 0.001
    t = 0.
    for ti in range(10000000):
        t = ti * dt
        z += RK4(dt, z, f_gravity)
        if ti % 100 == 0:
            xs = np.append(xs, z[:2,0])
        if ti % 1000 == 0:

            yield t, z, xs
#            print(t)
            
def func(data):
    t, z, xs = data
    x_history = xs.reshape((xs.size//2,2)).T
    nmax = 2
    x = z[:2]
    v = z[2:]
    R = np.array([[np.cos(t),-np.sin(t)],
                   [np.sin(t),np.cos(t)]])
    _x = np.linspace(-nmax,nmax,100)
    _y = _x.copy()
    X,Y = np.meshgrid(_x,_y)
#    X_rot = np.cos(t)*X + np.sin(t)*Y
#    Y_rot = -np.sin(t)*X + np.cos(t)*Y
#    x = R @ x
#    _r0 = R @ r0
#    _r1 = R @ r1
#    _L4 = R @ L4
    _r0, _r1, _L4, X_rot, Y_rot = r0, r1, L4, X, Y
    
    plt.clf()
    plt.title("t = {}".format(t))
#    plt.xlim(-nmax,nmax)
#    plt.ylim(-nmax,nmax)
    dx_range = 0.1
    plt.xlim(L4[0]-dx_range,L4[0]+dx_range)
    plt.ylim(L4[1]-dx_range,L4[1]+dx_range)
    plt.scatter(x[0],x[1], s=10)
    plt.plot(*x_history)
    plt.scatter(_L4[0],_L4[1], c="r", s=30)
    plt.scatter(_r0[0],_r0[1], c="b", s=30)
    plt.scatter(_r1[0],_r1[1], c="b", s=30)
    
    V = V_gravity(X_rot,Y_rot)
#    plt.contour(X,Y, V, C)
#    print("total E = {}".format(((z[2:,0]**2+z[2:,1])**2).sum()))
    
fig = plt.figure()
ani = animation.FuncAnimation(fig, func, gen,interval=10)
plt.show()