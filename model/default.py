"""Define model architecture here.
The model class must be a torch.nn.Module and should have forward method.

For custome model, they are needed to be treated accordingly in train function.
"""

import torch
import torch.nn as nn

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

class model(nn.Module):
    def __init__(self, input_shape=28*28, nr_classes=10):
        super(model, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape, 200)
        self.fc2 = nn.Linear(200, nr_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_shape) # Flatten the batched tensors
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        log_softmax_out = nn.LogSoftmax()(x)
        return log_softmax_out
        
    def forward_inference(self, x):
        raise NotImplementedError

    def get_name(self):
        return "default_model"

#################################### Test ######################################
def test():
    m = model()
    print(m)
    input_tensor = torch.empty((1,1,28,28))
    output = m(input_tensor)
    for out in output:
        print(out.size())
    print([(name, type(p)) for name, p in m.named_parameters()])
    print(m._modules['fc1'])

if __name__ == "__main__":
    test()