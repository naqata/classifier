from torch.utils.data import Dataset #, DataLoader
# from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
import os
from PIL import Image

# import matplotlib.pyplot as plt

class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        self.labels = []
        self.image_paths = []
        self.classes = []

        data_path = os.path.join(root, 'train' if train else 'test')
        self.classes = sorted([class_name for class_name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, class_name))])

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_path, class_name)
            for file in os.listdir(class_dir):
                path = os.path.join(class_dir, file)
                if os.path.isfile(path):
                    self.labels.append(label)
                    self.image_paths.append(path)

    def __getitem__(self, index):
        label = self.labels[index]
        image_path = self.image_paths[index]

        with Image.open(image_path) as img:
            image = img.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.labels)

# if __name__ == '__main__':
#     root_dir = '../animals'
#     image_size = (224, 224)
#     batch_size = 16
#     num_workers = 0

#     transform = Compose([
#         Resize(image_size),
#         ToTensor()
#     ])

#     train_set = AnimalDataset(root_dir, train=True, transform=transform)
    
#     print(len(train_set))
#     image, label = train_set[0]
    
#     # plt.imshow(image)
#     plt.imshow(ToPILImage()(image))
#     plt.title(f'Class: {train_set.classes[label]}')
#     plt.show()

    # train_loader = DataLoader(
    #     dataset=train_set,
    #     batch_size = batch_size,
    #     num_workers=num_workers,
    #     shuffle=True,
    #     drop_last = True
    # )

    # for iter, (images, labels) in enumerate(train_loader):
    #     print(images.shape)
    #     print(labels.shape)