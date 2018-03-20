# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 23:00:14 2016

@author: Yuki Miyake
"""


import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

plt.style.use("ggplot")

### 慣性質量mu
###  mu = M2 / (M1 + M2)
mu = 0.01 


x_unit = np.array([1.,0.])
r0 = mu * x_unit # 重い方(M2)の天体座標
r1 = (mu - 1) * x_unit # 軽い方(M1)の天体座標

L4 = np.array([mu - 1/2, np.sqrt(3) / 2])
L5 = np.array([mu - 1/2, -np.sqrt(3) / 2])
L2 = np.array([-(1 + (mu / 3)**(1/3)), 0])
L1 = np.array([-(1 - (mu / 3)**(1/3)), 0])
L3 = np.array([1 + 5 * mu / 12, 0])
#L4 = (r0 + r1) / 2 + np.array([0.,np.sqrt(3)/2])
Ls = [L1, L2, L3, L4, L5]
C = np.sort(np.array([3+3**(4/3)*mu**(2/3) - 10*mu/3,
              3+3**(4/3)*mu**(2/3) - 14*mu/3,3+mu,3-mu]))

################ Problem Setting #######################

#ラグランジュ点を指定。
x_base = L5


# 質点の個数
N =100

# 慣性系かどうか。 （True: 慣性系, False: 回転座標系）
rotate = False

########################################################

print("criteria = {}".format(mu*25.96))
def f_gravity(z):
    """質点の位置と速度を並べた行列zの時間微分dz/dt。
    	z = [
    		[x1,x2,...,xn],
    		[y1,y2,...,yn],
    		[vx1,vx2,...,vxn],
    		[vy1,vy2,...,vyn]
    		]
    """
    f = np.empty_like(z)
    f_x = f[:2]
    f_v = f[2:]
    x = z[:2]
    v = z[2:]
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
    ### ラグランジュ点まわりに正規分布ノイズを加えた小天体を配置。
    x = np.zeros((2,N)) + x_base[:,np.newaxis] + np.random.randn(2,N) * 0.01
    v = np.zeros_like(x)
    z = np.vstack((x,v))
    dt = 0.01
    t = 0.
    for ti in range(10000000):
        t = ti * dt
        z += RK4(dt, z, f_gravity)
        if ti % 10 == 0:

            yield t, z
            
def func(data):
    t, z = data

    nmax = 2
    x = z[:2]
    R = np.array([[np.cos(t),-np.sin(t)],
                   [np.sin(t),np.cos(t)]])
    
    if rotate:
        # 回転座標にする処理
        _x = R @ x
        _r0 = R @ r0
        _r1 = R @ r1
        _x_base = R @ x_base
    else:
        _x, _r0, _r1, _x_base = x, r0, r1, x_base
    
    ax.cla()
    ax.set_aspect("equal")
    
    ax.set_title("t = {:.3f}".format(t))
    ax.set_xlim(-nmax,nmax)
    ax.set_ylim(-nmax,nmax)
    
    ax.set_yticks(np.arange(-2,3))
    ax.scatter(_x[0],_x[1], s=2, marker=".")
    ax.scatter(_x_base[0],_x_base[1], c="black", s=6)
    ax.scatter(_r0[0],_r0[1], c="orange", s=30)
    ax.scatter(_r1[0],_r1[1], c="blue", s=6)
    
fig = plt.figure()
ax = fig.add_subplot(111)
ani = animation.FuncAnimation(fig, func, gen, interval=10)
plt.show()