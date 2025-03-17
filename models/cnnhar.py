import torch.nn as nn
import torch.nn.functional as F
        
        
class CNNHAR(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_channel=1, num_classes=6):
        super(CNNHAR, self).__init__()
        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_channel, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(35136, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        out = self.classifier(x)

        return out
        
class CNNHAR_header(nn.Module):
    def __init__(self, input_channel=1, num_classes=6):
        super(CNNHAR_header, self).__init__()
        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_channel, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(35136, 128),
            nn.ReLU(),
            nn.Dropout(),
            #nn.Linear(128, num_classes),
            )

    def forward(self, x):
        #print(x.shape)
        x = self.features(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        out = self.classifier(x)

        return out