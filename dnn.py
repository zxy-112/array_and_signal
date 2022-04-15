import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import aray
from utils import *

to_two = lambda x: np.concatenate((np.real(x)[np.newaxis, ...], np.imag(x)[np.newaxis, ...]), axis=0)
identity = lambda x: x
normalize = lambda x: x / np.max(x)
combiner = lambda f, g: lambda x: f(g(x))

to_two = combiner(normalize, to_two)
identity = combiner(normalize, identity)

####################
#######MODEL########
####################
class ArrayDataset(Dataset):
    def __init__(self, folder_name='coherent_data', offset=0, length=10000, transform=to_two, target_transform=to_two):
        self.path_name = os.path.join(os.path.expanduser('~'), folder_name)
        self.offset = offset
        self.length = length
        self.transform = combiner(torch.from_numpy, transform)
        self.target_transform = combiner(torch.from_numpy, target_transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_path = os.path.join(self.path_name, 'x'+str(index)+'.npy')
        y_path = os.path.join(self.path_name, 'y'+str(index)+'.npy')

        def load_func(path):
            with open(path, 'rb') as f:
                res = np.load(f)
            return res

        x = load_func(x_path)
        y = load_func(y_path)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

big_plan = {
    '11' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M'],  # , 512, 512,           'M'],
    '13' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M'],  # , 512, 512,           'M'],
    '16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M'],  # , 512, 512, 512,      'M'],
    '19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M']  # , 512, 512, 512, 512, 'M']
}
def make_layers(lis):
    res = []

    def helper(begin_index, input_chanel):
        if begin_index >= len(lis):
            return
        if lis[begin_index] == 'M':
            res.append(nn.MaxPool2d(kernel_size=2, stride=2))
            helper(begin_index+1, input_chanel)
        else:
            res.append(nn.Conv2d(input_chanel, lis[begin_index], kernel_size=3, padding=1))
            res.append(nn.BatchNorm2d(lis[begin_index]))
            helper(begin_index+1, lis[begin_index])
    helper(0, 2)

    return nn.Sequential(*res)

class CNN(nn.Module):
    def __init__(self, plan):
        super(CNN, self).__init__()

        self.feature_acquire = make_layers(plan)
        self.linear = nn.Conv2d(512, 16 * 2, kernel_size=1)

    def forward(self, x):
        output = self.feature_acquire(x)
        output = self.linear(output)
        output = output.view(output.shape[0], 2, 16, 16)

        return output

model = CNN(big_plan['19']).to(device)
model = model.double()
print(model)

####################
#######TRAIN########
####################
log_file_name = 'log.txt'
def write_log(message):
    print(message)
    with open(log_file_name, 'a') as f:
        f.write(message+'\n')

train_dataloader = DataLoader(ArrayDataset(length=9000), batch_size=64, shuffle=True)
test_dataloader = DataLoader(ArrayDataset(offset=9000, length=1000), batch_size=64)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 50

lucky_train, lucky_label = next(iter(train_dataloader))
print('train batch shape: \n', lucky_train.shape)
print('label batch shape: \n', lucky_label.shape)

to_one = lambda x: x[0][0] + x[0][1] * 1j
lucky_output = to_one(lucky_train)
lucky_cov = to_one(lucky_label)
lucky_ary = aray.UniformLineArray()
for _ in range(16):
    lucky_ary.add_element(aray.Element())

mvdr_weight = mvdr(lucky_output, 0, lucky_ary.steer_vector)
fig_ax = lucky_ary.response_plot(mvdr_weight)
right_weight = np.linalg.pinv(lucky_cov) @ lucky_ary.steer_vector(0)
lucky_ary.response_plot(right_weight, fig_ax_pair=fig_ax)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(x)
            write_log('loss: {:>7f}   [{:>5d}/{:>5d}]'.format(loss, current, size))

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y)

    test_loss = test_loss / num_batches
    write_log('Test loss: {:>8f}\n'.format(test_loss))

write_log(time.ctime())
for epoch in range(epochs):
    write_log('Epoch {}\n-----------------'.format(epoch+1))
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    torch.save(model, 'check_point{}.pth'.format(epoch+1))
write_log('Done')
