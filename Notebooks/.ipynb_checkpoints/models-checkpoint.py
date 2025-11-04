#!/usr/bin/env python
# coding: utf-8

from Imports import *
from Preprocessing import *
from Helper import *

X_train, y_train, X_test, y_test, X_val, y_val = Preprocess(df_train, df_test, df_val, balance = 'smote', classes = 'binary')

train_data = ClassificationDataset(X_train, y_train)
validation_data = ClassificationDataset(X_val, y_val)
test_data = ClassificationDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(validation_data, batch_size=64, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, drop_last=True)

X_batch, y_batch = next(iter(train_loader))

print("Batch X dtype:", X_batch.dtype)
print("Batch y dtype:", y_batch.dtype)

print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(validation_data))
print("Number of test samples:", len(test_data))

# Actual options here...
model = ""
n_epochs = 

train(model, n_epochs, lr, ...)
save(model)
load(model, savepoint)


model = model.to(device)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler_G = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)

# Empty array to track losses
losses = []

# Initialize metrics
bce = nn.BCELoss()
metric = accuracy
best_val_metric = 0
best_val_loss = 999

# Initialize a dictionary to store epoch-wise results
history = {
        'epoch': [],
        'train_loss': [],
        'train_metric': [],
        'val_loss': [],
        'val_metric': []
    }
