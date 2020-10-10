import torch
import numpy as np
import time

u1 = torch.tensor(0.1,requires_grad=True)
u2 = torch.tensor(0.0,requires_grad=True)
t = torch.tensor(1.0,requires_grad=True)

angle = torch.tensor(0.0)
v = torch.tensor(0.0)
x = torch.tensor(-30)
y = torch.tensor(5.0)
x_target = torch.tensor(1.0)
y_target = torch.tensor(-8.0)

# x = x + (v + u1*t)*torch.cos(angle+u2*t)*t
# y = y + (v + u1*t)*torch.sin(angle+u2*t)*t
# f = open('/Users/cedricxing/Desktop/loss.txt','w')
learning_step = 0.00001

def satisfyConstraints(x,y,u1,u2,t):
    # if u1 < -0.3 or u1 > 0.3:
    #     print('u1 not satisfy')
    #     return False
    # if angle < -np.pi / 2:
    #     return False
    # if u2 < -np.pi/6 or u2 > np.pi/6:
    #     print('u2 not satisfy')
    #     return False
    # if x < 10 and y < 10:
    #     print('x,y not satisfy, x:'+str(x)+' y:'+str(y))
    #     return False
    # if y > 35:
    #     print('x,y not satisfy, x:'+str(x)+' y:'+str(y))
    #     return False
    return True

def Forward(x0,y0,v0,angle0,t0,target0):
    u1 = torch.tensor(0.1,requires_grad=True)
    u2 = torch.tensor(0.0,requires_grad=True)
    t = torch.tensor(0.5,requires_grad=True)
    x_ = torch.tensor(x0.item())
    y_ = torch.tensor(y0.item())
    v_ = torch.tensor(v0.item())
    angle_ = torch.tensor(angle0.item())
    u1_ = torch.tensor(0.0)
    u2_ = torch.tensor(0.0)
    t_ = torch.tensor(1.0)
    pre_value = -1
    step = 0
    while True:
        v = v0 + u1 * t
        angle = angle0 + u2 * t
        x = x0 + torch.cos(angle)*(v0*t+0.5*u1*t*t)
        y = y0 + torch.sin(angle)*(v0*t+0.5*u1*t*t)
        target = (x-x_target)*(x-x_target)+(y-y_target)*(y-y_target)
        target.backward(retain_graph=True)
        temp = target.item()
        # print('x: %5f y: %5f'%(x.item(), y.item()))
        # print('Forward:' + str(temp))
        # print('u1.grad: ' + str(u1.grad.item()))
        if not satisfyConstraints(x,y,u1,u2,t) or t < 0:
            t.data = torch.tensor(-1.0)
            return x0,y0,v0,angle0,u1,u2,t,target0
            # return x_,y_,v_,angle_,u1_,u2_,t_
        if pre_value != -1 and temp + 0.0001 > pre_value:
            break
        pre_value = temp
        u1_.data = u1.data
        u2_.data = u2.data
        t_.data = t.data 
        u1.data = u1.data - u1.grad * learning_step
        if u1.data > 0.5:
            u1.data = torch.tensor(0.5)
        if u1.data < -0.5:
            u1.data = torch.tensor(-0.5)
        t.data = t.data - t.grad * learning_step
        u1.grad.data.zero_()
        t.grad.data.zero_()
        x_.data = x.data
        y_.data = y.data
        v_.data = v.data
        angle_.data = angle.data
    if target.item() < target0.item():
        return x,y,v,angle,u1,u2,t,target
    else:
        t.data = torch.tensor(-1.0)
        return x0,y0,v0,angle0,u1,u2,t,target0

def Turn(x0,y0,v0,angle0,t0,target0):
    learning_step = 0.00001
    u1 = torch.tensor(0.0,requires_grad=True)
    u2 = torch.tensor(-0.5,requires_grad=True)
    t = torch.tensor(0.5,requires_grad=True)
    x_ = torch.tensor(x0.item())
    y_ = torch.tensor(y0.item())
    v_ = torch.tensor(v0.item())
    angle_ = torch.tensor(angle0.item())
    u1_ = torch.tensor(0.0)
    u2_ = torch.tensor(0.0)
    t_ = torch.tensor(1.0)
    pre_value = -1
    while True:
        v = v0 + u1 * t
        angle = angle0 + u2 * t
        x = x0 + v0/u2*(torch.sin(angle)-torch.sin(angle0))
        y = y0 + v0/u2*(torch.cos(angle0) - torch.cos(angle))
        target = (x-x_target)*(x-x_target)+(y-y_target)*(y-y_target)
        target.backward(retain_graph=True)
        temp = target.item()
        # print('x: %5f y: %5f'%(x.item(), y.item()))
        # print('u2.grad: ' + str(u2.grad.item()))
        # print('Turn:' + str(temp))
        if not satisfyConstraints(x,y,u1,u2,t) or t < 0:
            print('not satisfy')
            t.data = torch.tensor(-1.0)
            return x0,y0,v0,angle0,u1,u2,t,target0
            # return x_,y_,v_,angle_,u1_,u2_,t_
        if pre_value != -1 and temp + 0.0001 > pre_value:
            break
        pre_value = temp
        u1_.data = u1.data
        u2_.data = u2.data
        t_.data = t.data
        u2.data = u2.data - u2.grad * learning_step
        t.data = t.data - t.grad * learning_step
        if u2.data < -np.pi/3:
            u2.data = torch.tensor(-np.pi/3)
        
        u2.grad.data.zero_()
        t.grad.data.zero_()
        x_.data = x.data
        y_.data = y.data
        v_.data = v.data
        angle_.data = angle.data
        
    if target.item() < target0.item():
        return x,y,v,angle,u1,u2,t,target
    else:
        t.data = torch.tensor(-1.0)
        return x0,y0,v0,angle0,u1,u2,t,target0

def print_info(x,y,v,angle,u1,u2,t):
    print('x: ' + str(x))
    print('y: ' + str(y))
    print('v: ' + str(v))
    print('angle: ' + str(angle))
    print('u1:' + str(u1))
    print('u2:' + str(u2))
    print('t: ' + str(t))

mode = 'Forward'
flag = True
f = open('/Users/admin/Desktop/result.txt','w')
time_start = time.time()
target = torch.tensor(10000000.0)

while pow(x-x_target.item(),2) + pow(y-y_target.item(),2) > 1:
    if mode == 'Forward':
        print('Forward:')
        x,y,v,angle,u1,u2,t,target = Forward(x,y,v,angle,t,target)
        mode = 'Turn'
        if x_target.item() == 0.0:
            x_target.data = torch.tensor(3.0)
        if t.item() == -1:
            continue
        f.write('1,&%d&%f&%f&%f&%f\n'%(t/0.01,u2,u1,x,y))
        print_info(x,y,v,angle,u1,u2,t)

    elif mode == 'Turn':
        print('Turn:')
        x,y,v,angle,u1,u2,t,target = Turn(x,y,v,angle,t,target)
        mode = 'Forward'
        if t.item() == -1:
            continue
        f.write('2,&%d&%f&%f&%f&%f\n'%(t/0.01,u2,u1,x,y))
        print_info(x,y,v,angle,u1,u2,t)
        # break
print('finished')
time_end = time.time()
print('time_cost: ' + str(time_end-time_start) + 's')
