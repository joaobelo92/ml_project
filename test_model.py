import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

model = torch.load('model.pt')
model.eval()
data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = 'data/real/object/'
image_dataset = datasets.ImageFolder(data_dir, data_transforms)
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4,
                                         shuffle=True, num_workers=4)
dataset_size = len(image_dataset)
class_names = ['bed', 'chair', 'drawer', 'lack', 'lamp']
# class_names = ['step_1', 'step_2', 'step_3', 'step_4', 'step_5']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# method that allows us to visualize the augmented data
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


visualize_model(model)
