import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Load data
transform = transforms.Compose(
    [torchvision.transforms.Resize(32), transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                          shuffle=True)
print("Finish loading data")


# Implement VGG11
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)

        # Batch
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(128)
        self.batch3 = nn.BatchNorm2d(256)
        self.batch4 = nn.BatchNorm2d(512)

        # DropOut
        self.drop = nn.Dropout(0.5)

        # Pool
        self.pool = nn.MaxPool2d(2, 2)

        # FC
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.batch1(self.conv1(x))))
        x = self.pool(F.relu(self.batch2(self.conv2(x))))
        x = F.relu(self.batch3(self.conv3(x)))
        x = self.pool(F.relu(self.batch3(self.conv4(x))))
        x = F.relu(self.batch4(self.conv5(x)))
        x = self.pool(F.relu(self.batch4(self.conv6(x))))
        x = F.relu(self.batch4(self.conv6(x)))
        x = self.pool(F.relu(self.batch4(self.conv6(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

net = Net()

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Training
for epoch in range(5):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("Epoch ", epoch + 1)
    # save model with epoch 3 - 5
    if epoch >= 2:
        PATH = './trained_epoch' + str(epoch + 1) + '.pth'
        torch.save(net.state_dict(), PATH)

print('Finished Training')

