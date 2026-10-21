# Source: fundamentals/Perceptron&Linear.ipynb · cell 17
# Related learning code; original stock MLP not recovered.
# Viewing excerpt; see original notebook for context.

class ZachLinear(nn.Module):
    def __init__(self,in_features, out_features):
        super(ZachLinear, self).__init__()
        self.weights = nn.Parameter(data=torch.rand(in_features, out_features))
        self.bias = nn.Parameter(data=torch.rand(out_features))
    def forward(self,x):
        x = torch.matmul(x,self.weights)
        wbias = torch.add(x,self.bias)
        # print("X:",wbias, wbias.shape)
        return wbias
class blobsClassifier(nn.Module):
    def __init__(self):
        super(blobsClassifier, self).__init__()
        self.model = nn.Sequential(
            ZachLinear(in_features=2,out_features=10),
            nn.ReLU(),
            ZachLinear(in_features=10, out_features=10),
            nn.ReLU(),
            ZachLinear(in_features=10, out_features=3),
        )
    def forward(self, x):
        return self.model(x)
blobsClassification = blobsClassifier().cpu()
