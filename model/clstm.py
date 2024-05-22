'''
based on implementation from https://github.com/automan000/Convolution_LSTM_pytorch
'''

import torch
import torch.nn as nn
from torch.autograd import Variable




class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, conv_stride, ):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.conv_stride = conv_stride
      

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, self.conv_stride, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        #print("x shape ", x.shape, " h shape: ", h.shape, " c shape: ", c.shape)
        #print("wci shape: ",self.Wxi(x).shape)
        #print("whi shape: ", self.Whi(x).shape)
        #print("self wci op shape: ",(c * self.Wci).shape)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(torch.device('cuda'))
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(torch.device('cuda'))
            self.Wco = Variable(torch.zeros(1, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(torch.device('cuda'))
        else:
            assert shape[0]//self.conv_stride == self.Wci.size()[2], 'Input Height Mismatched! %d vs %d' %(shape[0]//self.conv_stride, self.Wci.size()[2])
            assert shape[1]//self.conv_stride == self.Wci.size()[3], 'Input Width Mismatched!'
        #print("returning init h of size ", batch_size, hidden, shape[0], shape[1])cd
        return (Variable(torch.zeros(batch_size, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(torch.device('cuda')),
                Variable(torch.zeros(batch_size, hidden, shape[0]//self.conv_stride, shape[1]//self.conv_stride)).to(torch.device('cuda')))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, conv_stride,
                 pool_kernel_size=(2,2), step=1, effective_step=[1],
                 batch_normalization=True, dropout=0,):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.pool_kernel_size = pool_kernel_size
        self.conv_stride = conv_stride
        self.mp = nn.MaxPool2d(kernel_size=self.pool_kernel_size)
        self.batch_norm = batch_normalization
        self.dropout_rate=dropout
     
        #to be pool_size=2, strides=None, padding='valid', data_format='channels_last')
        #kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(hidden_channels[i], eps=1e-05, momentum=0.1, affine=True)
            for i in range(self.num_layers)])
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.conv_stride)
            setattr(self, name, cell)
           
            
            self._all_layers.append(cell)
           

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[:,step,:,:,:]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                #print("on cell step ", i ,"with name: ",name)
                #print("x size is:", x.size())
                if step == 0:
                    bsize, channels, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height,width))
                                                             #shape=(int(height/max(1,(self.pool_kernel_size[0]*(i)))),\
                                                             #       int(width/max(1,(self.pool_kernel_size[1]*(i))))))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                #print("h size is: ", h.size())
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
                
                if(self.dropout_rate):
                    x = self.dropout(x)
                if(self.batch_norm):
                    x = self.bn_layers[i](x)
                x = self.mp(x)
                
                
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        #print("returning shape: ", len(outputs), outputs[-1].shape)
        new_c=self.mp(new_c)
        return outputs, (x, new_c)


if __name__ == '__main__':
    # gradient check

    convlstm = ConvLSTM(input_channels=3, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, conv_stride=1,
                        pool_kernel_size=(2, 2), step=14, effective_step=[13])
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(2, 14, 3, 244, 244))
    target = Variable(torch.randn(2, 32, 244, 244)).double()

    output = convlstm(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
    print(res)
