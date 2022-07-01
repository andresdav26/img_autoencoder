import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()        
        # N, 1, 40, 40
        self.encoder = nn.Sequential(
            # nn.Dropout2d(0.05),
            nn.Conv2d(1, 32, 3, stride=1, padding=1), 
            nn.ReLU(),

            # nn.Dropout2d(0.05),
            nn.Conv2d(32, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(64, 32, 1, stride=1, padding=0), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),

            # nn.Dropout2d(0.05),
            nn.Conv2d(32, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), 
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.05),
            nn.Conv2d(64, 32, 1, stride=1, padding=0), 
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # nn.Dropout2d(0.05),
            nn.Conv2d(32, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(128, 32, 1, stride=1, padding=0), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),

            # nn.Dropout2d(0.05),
            nn.Conv2d(32, 128, 1, stride=1, padding=0),
            nn.BatchNorm2d(128),  
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),  
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(128, 32, 1, stride=1, padding=0),
            nn.BatchNorm2d(32), 
            nn.ReLU(),

            # nn.Dropout2d(0.05),
            nn.Conv2d(32, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),  
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),  
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(256, 32, 1, stride=1, padding=0),
            nn.BatchNorm2d(32), 
            nn.ReLU(),

            # nn.Dropout2d(0.05),
            nn.Conv2d(32, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),  
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),  
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(256, 32, 1, stride=1, padding=0),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
        )
        # N, 32, 40, 40
        self.decoder = nn.Sequential(
            # nn.Dropout2d(0.05),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),  
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),  
            nn.LeakyReLU(),
            # nn.Dropout2d(0.05),
            nn.Conv2d(16, 1, 3, stride=1, padding=1), 
            nn.LeakyReLU(),
        )

    def forward(self, x, device):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        T0 = torch.tensor([0]).to(device)
        T1 = torch.tensor([1]).to(device)
        output = T1 - torch.maximum(T0, (T1 - torch.maximum(T0,decoded)))
        return output