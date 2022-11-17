# https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import matplotlib.pyplot as plt

# load mnist
batch_size = 64
train_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data',
    train=True,
    download = True,
    transform = transforms.Compose(
        [transforms.ToTensor()])
     ),
     batch_size=batch_size
)

test_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data',
    train=False,
    transform=transforms.Compose(
    [transforms.ToTensor()])
    ),
    batch_size=batch_size)

def check_data(d): 
    t_sum = torch.round(torch.sum(d), decimals = 4)
    t_max = torch.round(torch.max(d), decimals = 4)
    t_min = torch.round(torch.min(d), decimals = 4)
    t_mean = torch.round(torch.mean(d), decimals = 4)
    print(f"sum: {t_sum}\nmax: {t_max}\nmin: {t_min}\nmean: {t_mean}")

x = nn.Parameter(torch.tensor([[0., 0.1, 0.2]]))
x.mean().backward()
x
# define model
class RBM(nn.Module):
   def __init__(self,
               n_vis=784, 
               n_hin=500,
               k=5): 
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis)) 
        self.h_bias = nn.Parameter(torch.zeros(n_hin)) 
        self.k = k 
    
   def sample_from_p(self,p):
       return F.relu(torch.sign(p - Variable(torch.rand(p.size())))) 
    
   def v_to_h(self,v):
        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
   def h_to_v(self,h):
        p_v = torch.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
   def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_) # what does this do?
        
        return v,v_
    
   def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

# run model 
rbm = RBM() # initialize class
print(rbm) #hmm, should show layes, shapes, parameters..
train_op = optim.SGD(rbm.parameters(), 0.1) # stochastic gradient descent with learning rate = 0.1

for epoch in range(10): # n epochs
    loss_ = []
    lost_back_ = []
    
    for _, (data,target) in enumerate(train_loader):
        data = Variable(data.view(-1,784))
        sample_data = data.bernoulli() # extract bernoulli dist. 
        
        v,v1 = rbm(sample_data) # feed data to module
        loss = rbm.free_energy(v) - rbm.free_energy(v1) # loss = difference in energy
        loss_.append(loss.data) # append to list of loss
        train_op.zero_grad() # 
        loss.backward() # backpropagation
        train_op.step() # take a step (grad * lr)
        for name, param in rbm.named_parameters(): 
            print(name)
            print(param.shape)
            check_data(param)
                
    print("Training loss for {} epoch: {}".format(epoch, np.mean(loss_)))

def show_adn_save(file_name,img):
    npimg = np.transpose(img.numpy(),(1,2,0))
    f = "./%s.png" % file_name
    plt.imshow(npimg)
    plt.imsave(f,npimg)

show_adn_save("real",make_grid(v.view(32,1,28,28))) # the current set of data
show_adn_save("generate",make_grid(v1.view(32, 1, 28, 28))) # what we have to start with 

#### going through step-by-step ####
# going through the code one step at a time
n_vis = 784 # has to be the total number
n_hin = 500
k = 1
W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2) # (500, 784) tensor from normal dist
v_bias = nn.Parameter(torch.zeros(n_vis)) # (784) tensor w. zeros
h_bias = nn.Parameter(torch.zeros(n_hin)) # (500) tensor w. zeros
k = k # number of repetitions

# extract one piece of data
for _, (data,target) in enumerate(train_loader):
    data = data

data.shape # (32, 1, 28, 28)
data_ = data.view(-1, 784)
data_.shape # (32, 784): 32 = half of batch-size?

### check values in data (rounding does not work at the moment)
def check_data(d): 
    t_sum = torch.round(torch.sum(d), decimals = 4)
    t_max = torch.round(torch.max(d), decimals = 4)
    t_min = torch.round(torch.min(d), decimals = 4)
    t_mean = torch.round(torch.mean(d), decimals = 4)
    print(f"sum: {t_sum}\nmax: {t_max}\nmin: {t_min}\nmean: {t_mean}")

check_data(data_)

### to bernoulli
sample_data = data_.bernoulli()
check_data(sample_data) # slightly different sum and mean, but very close 

### now we have the model
def sample_from_p(p):
    return F.relu(torch.sign(p - torch.rand(p.size())))

def v_to_h(v):
    p_h = torch.sigmoid(F.linear(v, W, h_bias))
    sample_h = sample_from_p(p_h)
    return p_h,sample_h

def h_to_v(h):
    p_v = torch.sigmoid(F.linear(h, W.t(), v_bias))
    sample_v = sample_from_p(p_v)
    return p_v,sample_v

def forward(v):
    pre_h1,h1 = v_to_h(v) # first we take the data (v)
    h_ = h1 # h_ = sample_h 
    for _ in range(k): # for range in k (k=1)
        pre_v_,v_ = h_to_v(h_) 
        pre_h_,h_ = v_to_h(v_)
    
    return v,v_

def free_energy(v):
    vbias_term = v.mv(v_bias) # shape: 32
    wx_b = F.linear(v, W, h_bias) # shape: 32, 500
    hidden_term = wx_b.exp().add(1).log().sum(1) # shape: 32
    return (-hidden_term - vbias_term).mean()

mat = torch.tensor([[1, 2], [3, 4]])
vec = torch.tensor([2, 4])
mat.mv(vec)
mat
loss_ = []

# running one full step #
## first print the parameters of the model 
for name, param in rbm.named_parameters(): 
    print(f"{name:-^20}")
    print(param.shape)
    check_data(param)
 
## forward is where it happens
x = v.mv(v_bias)
x.shape
v.shape
v, v1 = forward(sample_data)
check_data(v) # sample data
check_data(v1) # almost = v1 (stochastic noise) 

## free energy
v_free = free_energy(v) # free energy in v
v1_free = free_energy(v1) # free energy in v1
loss = v_free - v1_free # what we want to minimize
print(f"loss: {loss:-^30}")

## training step
loss_.append(loss.data) # append to list of loss
train_op.zero_grad() # 
loss.backward() # backpropagation
train_op.step() # 

## figure out what all of the torch stuff does: 
### torch.rand(): returns tensor with random numbers from uniform dist [0, 1)
torch.rand(4)

### torch.sign(): returns tensor with sign (positive, negative, neutral). 
torch.sign(torch.tensor([-3.1, 100, 0])) # -1, 1, 0

### torch.randn(): tensor with random numbers from normal distribution 
### with mean=0 and variance=1 (standard normal distribution)
x = torch.randn(2, 3)

### torch.zeros(): returns tensor with zeros
torch.zeros(10)

### nn.Parameter(): converts it to different kind of tensor (module parameter)
nn.Parameter(x)

### Variable(): this is deprecated.

### F.linear(): applies linear transformation: y = xA^t + b

### F.relu(): applies relu element-wise
F.relu(torch.tensor([-3, 0, 100])) # (0, 0, 100)

#### other problems ####
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") # cpu

model = RBM().to(device)
print(model)

## three parameters 
# W: 500, 784
# v_bias: 784
# h_bias: 500
# perhaps these change with updating...?
for name, param in rbm.named_parameters(): 
    print(name)
    print(param.shape)
    check_data(param)




### testing the linear thing
input_x = torch.tensor([[0, 1, 1], [1, 0, 0], [0, 1, 1]]) 
weight_x = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 1]])

F.linear(input = input_x, weight = weight_x.T, bias = bias)
bias = torch.tensor([2, 3])

weight_x.shape
input_x.shape

input_x


F.linear(v,self.W,self.h_bias)