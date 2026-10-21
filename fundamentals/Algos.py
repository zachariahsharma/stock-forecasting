import torch
from torch import nn
class ZachPerceptron(nn.Module):
    def __init__(self, in_features):
        super(ZachPerceptron, self).__init__()
        self.weights = nn.Parameter(data=torch.rand(in_features, 1))
        self.bias = nn.Parameter(data=torch.rand(1,1))
    def forward(self,x):
        return x @ self.weights + self.bias
class ZachLinear(nn.Module):
    def __init__(self,in_features, out_features, bias=True):
        super(ZachLinear, self).__init__()
        self.isbias = bias
        self.weights = nn.Parameter(data=torch.rand(in_features, out_features))
        self.bias = nn.Parameter(data=torch.rand(out_features))
        nn.init.xavier_uniform_(self.weights)
    def forward(self,x):
        x = torch.matmul(x,self.weights)
        if self.isbias:
            wbias = torch.add(x,self.bias)
        else:
            wbias = x
        return wbias
class ZachRNN(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(ZachRNN, self).__init__()
        self.hidden_state = hidden_size
        self.inputWeight = nn.Parameter(torch.randn(input_size, hidden_size))
        self.hiddenWeight = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.outputWeight = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.biasOutput = nn.Parameter(torch.randn(hidden_size))
        self.activation = nn.Tanh()
        nn.init.xavier_uniform_(self.inputWeight)
        nn.init.xavier_uniform_(self.hiddenWeight)
        nn.init.xavier_uniform_(self.outputWeight)
    def forward(self,x):
        batch, sequenceLength, _ = x.shape
        hidden = torch.zeros(batch, self.hidden_state).to(x.device)
        outputs = []
        for i in range(sequenceLength):
            input_current = x[:, i, :]  
            combined = torch.add(
                torch.matmul(input_current, self.inputWeight) + torch.matmul(hidden, self.hiddenWeight),
                self.bias
            ) 
            hidden = self.activation(combined)
            output = torch.add(torch.matmul(hidden, self.outputWeight), self.biasOutput)
            outputs.append(output)
        return torch.stack(outputs, dim=1), hidden
class ZachLSTM(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(ZachLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.inputWeights = nn.Parameter(data=torch.randn(input_size, hidden_size*4))
        self.shortTermHiddenWeights = nn.Parameter(data=torch.randn(hidden_size, hidden_size*4))
        self.bias = nn.Parameter(data=torch.randn(hidden_size*4))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.inputWeights)
        nn.init.xavier_uniform_(self.shortTermHiddenWeights)
        nn.init.zeros_(self.bias)
    def forward(self,x):
        batch_size, sequenceLength, _ = x.shape
        shortHidden = torch.zeros(batch_size, self.hidden_size)
        longHidden = torch.zeros(batch_size,self.hidden_size)
        outputs = []
        for i in range(sequenceLength):
            inputCurrent = x[:, i, :]
            inputsPostWeight = inputCurrent @ self.inputWeights
            shortHiddenPostWeight = shortHidden @ self.shortTermHiddenWeights
            combined = inputsPostWeight + shortHiddenPostWeight + self.bias
            forgetGateInput, percentageInputInput, LogitInputInput, percentageOutputInput = torch.split(combined, self.hidden_size, dim=1)
            forgetGateInput = self.sigmoid(forgetGateInput)
            percentageInputInput = self.sigmoid(percentageInputInput)
            LogitInputInput = self.tanh(LogitInputInput)
            percentageOutputInput = self.sigmoid(percentageOutputInput)
            longHidden = longHidden * forgetGateInput
            longHidden = longHidden + (percentageInputInput * LogitInputInput)
            shortHidden = self.tanh(longHidden) * percentageOutputInput
            outputs.append(shortHidden)
        outputs = torch.stack(outputs, dim=1)
        return outputs, (shortHidden, longHidden)

class ZachGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ZachGRU, self).__init__()
        self.hidden_size = hidden_size
        self.inputWeights = nn.Parameter(data=torch.randn(input_size, hidden_size*3))
        self.hiddenWeights = nn.Parameter(data=torch.randn(hidden_size,hidden_size*3))
        self.bias = nn.Parameter(data=torch.randn(hidden_size*3))
        nn.init.xavier_uniform_(self.inputWeights)
        nn.init.xavier_uniform_(self.hiddenWeights)
        nn.init.zeros_(self.bias)
    def forward(self,x):
        batchLength, sequenceLength, _ = x.shape
        hidden = torch.zeros(batchLength,self.hidden_size)
        outputs = []
        for i in range(sequenceLength):
            inputCurrent = x[:,i,:]
            combined = inputCurrent @ self.inputWeights + hidden @ self.hiddenWeights + self.bias
            updateGate, resetGate, candidate_pre = torch.split(combined, self.hidden_size, dim=1)
            resetGate = torch.sigmoid(resetGate)
            updateGate = torch.sigmoid(updateGate)
            hidden = hidden * updateGate
            hidden = hidden + torch.tanh(resetGate*hidden@ self.hiddenWeights[:, 2 * self.hidden_size:]+candidate_pre)*(1-updateGate)
            outputs.append(hidden)
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden