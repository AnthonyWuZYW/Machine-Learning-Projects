import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import statistics
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [torchvision.transforms.Resize(32), transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)
print("Finish loading data")


# VGG11 Model
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


epochs = [3, 4, 5]
training_loss = []
training_accuracy = []
test_loss = []
test_accuracy = []

net = Net()

for epoch in epochs:
    # previous trained model
    PATH = './trained-models/trained_epoch' + str(epoch) + '.pth'
    net.load_state_dict(torch.load(PATH))

    criterion = nn.CrossEntropyLoss()

    net.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        losses = []
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)

            loss = criterion(outputs, labels)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()
            if i % 20 == 19:
                losses.append(running_loss / 20)
                running_loss = 0.0
    
        training_loss.append(statistics.mean(losses))
        training_accuracy.append(100 * correct / total)

        correct = 0
        total = 0
        losses = []
        running_loss = 0.0
        for i, data in enumerate(testloader, 0):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)

            loss = criterion(outputs, labels)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()
            if i % 20 == 19:
                losses.append(running_loss / 20)
                running_loss = 0.0
    
        test_loss.append(statistics.mean(losses))
        test_accuracy.append(100 * correct / total)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_figheight(10)
fig.set_figwidth(12)

ax1.plot(epochs, training_loss)
ax1.set_title("training loss vs the number of epochs")
ax1.set_xlabel("the number of epochs")
ax1.set_ylabel("training loss")

ax2.plot(epochs, training_accuracy, 'tab:orange')
ax2.set_title("training accuracy vs the number of epochs")
ax2.set_xlabel("the number of epochs")
ax2.set_ylabel("training accuracy (%)")

ax3.plot(epochs, test_loss, 'tab:green')
ax3.set_title("test loss vs the number of epochs")
ax3.set_xlabel("the number of epochs")
ax3.set_ylabel("test loss")

ax4.plot(epochs, test_accuracy , 'tab:red')
ax4.set_title("test accuracy vs the number of epochs")
ax4.set_xlabel("the number of epochs")
ax4.set_ylabel("test accuracy (%)")

plt.show()