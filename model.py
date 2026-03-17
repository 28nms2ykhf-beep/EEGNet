# model.py
from torcheeg.models import EEGNet

def create_eegnet(chunk_size, num_electrodes, F1, F2, D, num_classes,
                  kernel_1, kernel_2, dropout):
    model = EEGNet(chunk_size=chunk_size,
                   num_electrodes=num_electrodes,
                   F1=F1,
                   F2=F2,
                   D=D,
                   num_classes=num_classes,
                   kernel_1=kernel_1,     
                   kernel_2=kernel_2,      
                   dropout=dropout)
    
    return model
