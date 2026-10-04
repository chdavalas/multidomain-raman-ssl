import torch.nn as nn
import torch.nn.functional as F

class Conv1DModel(nn.Module):
    def __init__(self, output_dim, input_channels, input_sequence_length):
        super(Conv1DModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv13 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv14 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv15 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv16 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the final sequence length
        final_sequence_length = self.calculate_final_length(input_sequence_length)
        flattened_size = 512 * final_sequence_length

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, output_dim)

        self.dropout = nn.Dropout(0.5)

    def calculate_final_length(self, initial_length):
        # Calculates the final sequence length after all convolutional and pooling layers.
        def apply_pooling(length, kernel_size, stride):
            return (length - kernel_size) // stride + 1

        length = initial_length
        # Apply each pooling operation
        length = apply_pooling(length, kernel_size=2, stride=2)  # pool1
        length = apply_pooling(length, kernel_size=2, stride=2)  # pool2
        length = apply_pooling(length, kernel_size=2, stride=2)  # pool3
        length = apply_pooling(length, kernel_size=2, stride=2)  # pool4
        length = apply_pooling(length, kernel_size=2, stride=2)  # pool5

        return length

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
