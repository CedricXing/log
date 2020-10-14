import torch
import numpy as np
import time

learning_step = 0.0001
M = 1.0
g = 10.0
L = 1.0
I = 40.0

T1 = torch.tensor(1.0,requires_grad = True)
T2 = torch.tensor(1.0,requires_grad = True)
T3 = torch.tensor(1.0,requires_grad = True)
t = torch.tensor(1.0,requires_grad = True)

angle = torch.tensor(0.0)
vx = torch.tensor(0.0)
vz = torch.tensor(10.0)
x = torch.tensor(0.0)
z = torch.tensor(0.0)
x_target = torch.tensor(10.0)
z_target = torch.tensor(0.0)

def satisfyConstraints(x,z,vx,vz,t):
    if x > 5 and x < 5.5 and (z < 5 or z > 7):
        return False

    return True

def Rise(x0,z0,vx0,vz0,angle0,t0,target0):
    T1 = torch.tensor(0.0,requires_grad=True)
    T2 = torch.tensor(M*g,requires_grad=True)
    ## range for T3 [0,2]
    T3 = torch.tensor(1.0,requires_grad=True)
    t = torch.tensor(1.0,requires_grad = True)

    x_ = torch.tensor(x0.item())
    z_ = torch.tensor(z0.item())
    vx_ = torch.tensor(vx0.item())
    vz_ = torch.tensor(vz0.item())
    angle_ = torch.tensor(angle0.item())

    T1_ = torch.tensor(0.0)
    T2_ = torch.tensor(M*g)
    T3_ = torch.tensor(1.0)
    t_ = torch.tensor(1.0)
    pre_value = -1

    while True:
        angle = angle0 + 0.5 * L / I * (T1 - T3) * t * t
        vx = vx0 + 1/M * (T1 + T2 + T3) * (angle0 * t + 1/6 * L / I * (T1 - T3) * t * t * t)
        vz = vz0 + 1/M * (T1 + T2 + T3) * (t - (angle0 * angle0 * t + 1/3*angle0*L/I*(T1-T3)*t*t*t+1/20*L*L/I*I*(T1-T3)*(T1-T3)*t*t*t*t*t) / 2) - g * t
        x = x0 + vx0*t + 1/M * (T1 + T2 + T3) * (0.5 * angle0 * t * t + 1 / 24 * L / I * (T1-T3) *t*t*t*t)
        z = z0 + vz0*t + 1/M * (T1 + T2 + T3) * (0.5 * t * t - (0.5 * angle0 * angle0 * t * t + 1/12 * angle0 * L/I * (T1-T3)*t*t*t*t+1/120*L*L/I*I*(T1-T3)*(T1-T3)*t*t*t*t*t*t) / 2) - 0.5 * g * t * t

        target = (x-x_target)*(x-x_target)+(z-z_target)*(z-z_target)
        target.backward(retain_graph=True)
        temp = target.item()

        if not satisfyConstraints(x,z,vx,vz,t) or t < 0:
            print('not satisfied\n')
            t.data = torch.tensor(-1.0)
            return x0,z0,vx0,vz0,angle0,T1,T2,T3,t,target0
        
        if pre_value != -1 and temp + 0.0001 > pre_value:
            break
        pre_value = temp
        T1_.data = T1.data
        T2_.data = T2.data
        T3_.data = T3.data
        t_.data = t.data
        T3.data = T3.data - T3.grad * learning_step

        if T3.data > 2.0:
            T3.data = torch.tensor(2.0)
        if T3.data < 0.0:
            T3.data = torch.tensor(0.0)
        
        t.data = t.data - t.grad * learning_step
        T3.grad.data.zero_()
        t.grad.data.zero_()
        x_.data = x.data
        z_.data = z.data
        vx_.data = vx.data
        vz_.data = vz.data
        angle_.data = angle.data
    if target.item() < target0.item():
        return x,z,vx,vz,angle,T1,T2,T3,t,target
    else:
        t.data = torch.tensor(-1.0)
        return x0,z0,vx0,vz0,angle0,T1,T2,T3,t,target0


def Cruise(x0,z0,vx0,vz0,angle0,t0,target0):
    T1 = torch.tensor(0.0,requires_grad=True)
    ## range for T2 [0,16]
    T2 = torch.tensor(8.0,requires_grad=True)
    T3 = torch.tensor(0.0,requires_grad=True)
    t = torch.tensor(1.0,requires_grad = True)
    

    x_ = torch.tensor(x0.item())
    z_ = torch.tensor(z0.item())
    vx_ = torch.tensor(vx0.item())
    vz_ = torch.tensor(vz0.item())
    angle_ = torch.tensor(angle0.item())

    T1_ = torch.tensor(0.0)
    T2_ = torch.tensor(8.0)
    T3_ = torch.tensor(0.0)
    t_ = torch.tensor(1.0)
    pre_value = -1

    while True:
        angle = angle0 + 0.5 * L / I * (T1 - T3) * t * t
        vx = vx0 + 1/M * (T1 + T2 + T3) * (angle0 * t + 1/6 * L / I * (T1 - T3) * t * t * t)
        vz = vz0 + 1/M * (T1 + T2 + T3) * (t - (angle0 * angle0 * t + 1/3*angle0*L/I*(T1-T3)*t*t*t+1/20*L*L/I*I*(T1-T3)*(T1-T3)*t*t*t*t*t) / 2) - g * t
        x = x0 + vx0*t + 1/M * (T1 + T2 + T3) * (0.5 * angle0 * t * t + 1 / 24 * L / I * (T1-T3) *t*t*t*t)
        z = z0 + vz0*t + 1/M * (T1 + T2 + T3) * (0.5 * t * t - (0.5 * angle0 * angle0 * t * t + 1/12 * angle0 * L/I * (T1-T3)*t*t*t*t+1/120*L*L/I*I*(T1-T3)*(T1-T3)*t*t*t*t*t*t) / 2) - 0.5 * g * t * t

        target = (x-x_target)*(x-x_target)+(z-z_target)*(z-z_target)
        target.backward(retain_graph=True)
        temp = target.item()

        if not satisfyConstraints(x,z,vx,vz,t) or t < 0:
            print('not satisfied\n')
            t.data = torch.tensor(-1.0)
            return x0,z0,vx0,vz0,angle0,T1,T2,T3,t,target0
        
        if pre_value != -1 and temp + 0.0001 > pre_value:
            break
        pre_value = temp
        T1_.data = T1.data
        T2_.data = T2.data
        T3_.data = T3.data
        t_.data = t.data
        T2.data = T2.data - T2.grad * learning_step

        if T2.data > 16.0:
            T3.data = torch.tensor(16.0)
        if T2.data < 0.0:
            T2.data = torch.tensor(0.0)
        
        t.data = t.data - t.grad * learning_step
        T2.grad.data.zero_()
        t.grad.data.zero_()
        x_.data = x.data
        z_.data = z.data
        vx_.data = vx.data
        vz_.data = vz.data
        angle_.data = angle.data
    if target.item() < target0.item():
        return x,z,vx,vz,angle,T1,T2,T3,t,target
    else:
        t.data = torch.tensor(-1.0)
        return x0,z0,vx0,vz0,angle0,T1,T2,T3,t,target0


def Dive(x0,z0,vx0,vz0,angle0,t0,target0):
    ## range for T1 [0,2]
    T1 = torch.tensor(1.0,requires_grad=True)
    T2 = torch.tensor(M*g,requires_grad=True)
    T3 = torch.tensor(0.0,requires_grad=True)
    t = torch.tensor(1.0,requires_grad = True)

    x_ = torch.tensor(x0.item())
    z_ = torch.tensor(z0.item())
    vx_ = torch.tensor(vx0.item())
    vz_ = torch.tensor(vz0.item())
    angle_ = torch.tensor(angle0.item())

    T1_ = torch.tensor(1.0)
    T2_ = torch.tensor(M*g)
    T3_ = torch.tensor(0.0)
    t_ = torch.tensor(1.0)
    pre_value = -1

    while True:
        angle = angle0 + 0.5 * L / I * (T1 - T3) * t * t
        vx = vx0 + 1/M * (T1 + T2 + T3) * (angle0 * t + 1/6 * L / I * (T1 - T3) * t * t * t)
        vz = vz0 + 1/M * (T1 + T2 + T3) * (t - (angle0 * angle0 * t + 1/3*angle0*L/I*(T1-T3)*t*t*t+1/20*L*L/I*I*(T1-T3)*(T1-T3)*t*t*t*t*t) / 2) - g * t
        x = x0 + vx0*t + 1/M * (T1 + T2 + T3) * (0.5 * angle0 * t * t + 1 / 24 * L / I * (T1-T3) *t*t*t*t)
        z = z0 + vz0*t + 1/M * (T1 + T2 + T3) * (0.5 * t * t - (0.5 * angle0 * angle0 * t * t + 1/12 * angle0 * L/I * (T1-T3)*t*t*t*t+1/120*L*L/I*I*(T1-T3)*(T1-T3)*t*t*t*t*t*t) / 2) - 0.5 * g * t * t

        target = (x-x_target)*(x-x_target)+(z-z_target)*(z-z_target)
        target.backward(retain_graph=True)
        temp = target.item()

        if not satisfyConstraints(x,z,vx,vz,t) or t < 0:
            t.data = torch.tensor(-1.0)
            return x0,z0,vx0,vz0,angle0,T1,T2,T3,t,target0
        
        if pre_value != -1 and temp + 0.0001 > pre_value:
            break
        pre_value = temp
        T1_.data = T1.data
        T2_.data = T2.data
        T3_.data = T3.data
        t_.data = t.data
        T1.data = T1.data - T1.grad * learning_step

        if T1.data > 8.0:
            T1.data = torch.tensor(8.0)
        if T1.data < 0.0:
            T1.data = torch.tensor(0.0)
        
        t.data = t.data - t.grad * learning_step
        T1.grad.data.zero_()
        t.grad.data.zero_()
        x_.data = x.data
        z_.data = z.data
        vx_.data = vx.data
        vz_.data = vz.data
        angle_.data = angle.data
    if target.item() < target0.item():
        return x,z,vx,vz,angle,T1,T2,T3,t,target
    else:
        t.data = torch.tensor(-1.0)
        return x0,z0,vx0,vz0,angle0,T1,T2,T3,t,target0

def print_info(x,z,vx,vz,angle,T1,T2,T3,t):
    print('x: ' + str(x))
    print('z: ' + str(z))
    print('vx: ' + str(vx))
    print('vz: ' + str(vz))
    print('angle: ' + str(angle))
    print('T1:' + str(T1))
    print('T2:' + str(T2))
    print('T3:' + str(T3))
    print('t: ' + str(t))

mode = 'Dive'
flag = True
f = open('/home/cedricxing/Desktop/result.txt','w')
time_start = time.time()
target = torch.tensor(10000000.0)

# while pow(x-x_target.item(),2) + pow(z-z_target.item(),2) > 1:
while pow(x-x_target.item(),2) > 1:
    if mode == 'Cruise':
        print('Cruise:')
        x,z,vx,vz,angle,T1,T2,T3,t,target = Cruise(x,z,vx,vz,angle,t,target)
        mode = 'Rise'
        if t.item() == -1:
            continue
        f.write('1,&%d&%f&%f&%f&%f&%f\n'%(t/0.01,T1,T2,T3,x,z))
        print_info(x,z,vx,vz,angle,T1,T2,T3,t)

    elif mode == 'Rise':
        print('Rise:')
        x,z,vx,vz,angle,T1,T2,T3,t,target = Rise(x,z,vx,vz,angle,t,target)
        mode = 'Dive'
        if t.item() == -1:
            continue
        f.write('2,&%d&%f&%f&%f&%f&%f\n'%(t/0.01,T1,T2,T3,x,z))
        print_info(x,z,vx,vz,angle,T1,T2,T3,t)

    elif mode == 'Dive':
        print('Dive:')
        x,z,vx,vz,angle,T1,T2,T3,t,target = Dive(x,z,vx,vz,angle,t,target)
        mode = 'Cruise'
        if t.item() == -1:
            continue
        f.write('3,&%d&%f&%f&%f&%f&%f\n'%(t/0.01,T1,T2,T3,x,z))
        print_info(x,z,vx,vz,angle,T1,T2,T3,t)
    
print('finished')
time_end = time.time()
print('time_cost: ' + str(time_end-time_start) + 's')