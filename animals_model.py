import torch
import torch.nn as nn
# from torchsummary import summary

class AnimalModel(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 224, 224)):
        super().__init__()
        self.input_shape = input_shape

        # Convolutional layes
        self.conv1 = self.make_block(in_channels=3, out_channels=16, num_layers=2)
        self.conv2 = self.make_block(in_channels=16, out_channels=32, num_layers=2)
        self.conv3 = self.make_block(in_channels=32, out_channels=64, num_layers=3)
        self.conv4 = self.make_block(in_channels=64, out_channels=128, num_layers=3)
        self.conv5 = self.make_block(in_channels=128, out_channels=256, num_layers=3)

        # Calculate the output size after convolutions
        in_features = self.get_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.fc_final = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1) # Flatten for fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc_final(x)
        return x
    
    @staticmethod # A static method does not receive an implicit first argument. // no 'self'
    def make_block(in_channels, out_channels, kernel_size=3, padding=1, num_layers=2):
        """
        Creates a block of convolutional layers followed by a max pooling layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
            padding (int, optional): Padding added to all four sides of the input. Default is 1.
            num_layers (int): Number of convolutional layers in the block.

        Returns:
            nn.Sequential: A sequential container of the convolutional block.
        """
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels # Update for next layer
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    def get_conv_output_size(self):
        with torch.no_grad():
            x = torch.empty(1, *self.input_shape) #Batch size 1 with input shape
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            return x.numel() # Total number of elements in the tensor

# if __name__ == '__main__':
#     num_classes = 10
#     input_shape = (3, 224, 224)
#     model = AnimalModel(num_classes=num_classes, input_shape=input_shape)
#     print(model)
#     summary(model, input_size=input_shape)

#     data = torch.rand(8, *input_shape)
#     print(data.shape)

#     output = model(data)
#     print(output.shape)
