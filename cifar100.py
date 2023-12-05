import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

CIFAR100_PATH = '/data/zhiyu/dataset'

class CIFAR100WithNoise(torchvision.datasets.CIFAR100):
    def __init__(self, noise_prob=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_prob = noise_prob
        if self.train:  # Only add noise in the training set
            self.add_noise()

    def add_noise(self):
        num_classes = 100
        for i in range(len(self.targets)):
            if np.random.rand() < self.noise_prob:
                self.targets[i] = np.random.choice([j for j in range(num_classes) if j != self.targets[i]])

def load_cifar100_with_noise(noise_prob=0.1, batch_size=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = CIFAR100WithNoise(root=CIFAR100_PATH, train=True, download=True, transform=transform_train, noise_prob=noise_prob)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=1)

    return trainloader, testloader


# # Example usage
# trainloader, testloader = load_cifar10_with_noise(noise_prob=0.1)
