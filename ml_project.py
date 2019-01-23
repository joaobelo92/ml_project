import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# torchvision and torch.utils.data are useful to load the data
# with around 120 training images and 75 validation images this is
# a small dataset to generalize, but using transfer learning we
# should be able to generalize reasonably well.

# data augmentation and normalization for training
# only necessary to normalize for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # crops image to random size and aspect ratio
        transforms.RandomHorizontalFlip(),  # Horizontally flip the image with given p (def: 0.5)
        transforms.ToTensor(),              # converts a PIL image or np array to a FloatTensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/stages'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_train_loss_plot = []
    epoch_val_loss_plot = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        # epoch training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase is 'train':
                epoch_train_loss_plot.append(epoch_loss)
            else:
                epoch_val_loss_plot.append(epoch_loss)
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} loss: {epoch_loss:.4f} epoch acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best val acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return (model, epoch_train_loss_plot, epoch_val_loss_plot)


# load a resnet and reset final fully connected layer
# model_ft = models.resnet50(pretrained=True)
# num_features = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_features, 5)

# load a vgg model and reset final fully connected layer
model_ft = models.vgg19(pretrained=True)
num_features = model_ft.classifier[6].in_features
features = list(model_ft.classifier.children())[:-1] 
features.extend([nn.Linear(num_features, 5)])
model_ft.classifier = nn.Sequential(*features)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.002, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=0.7, gamma=0.1)

num_epochs = 25
model_ft, epoch_train_loss_plot, epoch_val_loss_plot = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)

plt.plot(epoch_train_loss_plot, label='train')
plt.plot(epoch_val_loss_plot, label='val')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('stages_loss_plot.png')

torch.save(model_ft.state_dict(), 'stages_model_state_dict.pt')
torch.save(model_ft, 'stages_model.pt')
