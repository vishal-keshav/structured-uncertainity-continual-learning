import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

class model(nn.Module):
    def __init__(self, input_shape, nr_tasks, nr_classes):
        super(model, self).__init__()
        self.input_shape = input_shape
        self.nr_tasks = nr_tasks
        self.nr_classes = nr_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)
        self.classifier_heads = nn.ModuleList()
        for task_id in range(nr_tasks):
            self.classifier_heads.append(nn.Linear(288, nr_classes))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        outputs = []
        for task_id in range(self.nr_tasks):
            outputs.append(self.classifier_heads[task_id](x))
        log_softmax_outputs = []
        for task_id in range(self.nr_tasks):
            log_softmax_outputs.append(nn.LogSoftmax()(outputs[task_id]))
        return log_softmax_outputs
        
    def forward_inference(self, x):
        raise NotImplementedError

    def get_name(self):
        return "multi-headed-cnn"

#################################### Test ######################################
def test():
    m = model(28*28, 5, 2)
    print(m)
    input_tensor = torch.empty((1,1,28,28))
    output = m(input_tensor)
    for out in output:
        print(out.size())
    print([(name, type(p)) for name, p in m.named_parameters()])
    print(m._modules['conv2'])
    summary(m, (1, 28, 28))


if __name__ == "__main__":
    test()