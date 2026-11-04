
import torch
import torch.nn as nn


class SRCNN(nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=9,
            padding=4
        )

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=1
        )

        # Third convolution layer
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=3,
            kernel_size=5,
            padding=2
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))

        x = self.conv3(x)

        return x


if __name__ == "__main__":

    model = SRCNN()

    # Create a sample RGB image tensor
    sample_input = torch.rand(1, 3, 256, 256)

    # Test the model
    with torch.no_grad():
        output = model(sample_input)

    print("SRCNN Model created successfully!")
    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Total parameters:", total_parameters)
