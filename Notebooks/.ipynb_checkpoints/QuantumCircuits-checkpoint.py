def initialize(model, load_path = None, binary = True, multiclass = True):

    # Conditionals for all model types
    if model == "DNN":
        #Initialize architecture here
        model = 

    if model == "RandomLayer":
        #Initialize architecture here
        from RandomLayer import RandomLayerCircuit as QuantumFeatureEmbeddingBatch

    if model == "StronglyEntangled":
        #Initialize architecture here
        from StronglyEntangled import StronglyEntangledCircuit as QuantumFeatureEmbeddingBatch
        

    
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

    if load_path != None:
        from load import load
        load(model, load_path)
        
    return model

# ====              Model Architectures                  ====
# ===========================================================
# 8-feature Classical
# 40-feature Classical (RS 0.5 Correlation)
# 43-feature Classical (RS 3-Layer FE)
#
#############################################################
# Binary DNN - Complete Classical
class BinaryDNN_PureClassical(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        # Classifies based on encoded features
        self.classifier = nn.Sequential(

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded)

        return class_probs  # Shape: (batch_size, 1)

#############################################################
# Multiclass DNN - Complete Classical
class MulticlassDNN_PureClassical(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        # Classifies based on encoded features
        self.classifier = nn.Sequential(

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded)

        return class_probs  # Shape: (batch_size, 1)
        
#############################################################
# Binary DNN - Random Shadows 40-feature Input
class BinaryDNN_RS40(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encodes features from dataset
        self.feature_encoder = nn.Sequential(
            nn.Linear(40, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded)

        return class_probs  # Shape: (batch_size, 1)

#############################################################
# Multiclass DNN - Random Shadows 44-feature Input
class MulticlassDNN_RS44(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = nn.Sequential(
            nn.Linear(40, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_target = self.classifier(feats_encoded)

        return class_target  # Shape: (batch_size, 4)

#############################################################
# Binary DNN - Random Shadows 43-feature Input
class BinaryDNN_RS43(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = nn.Sequential(
            nn.Linear(43, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded)

        return class_probs  # Shape: (batch_size, 1)

#############################################################
# Multiclass DNN - Random Shadows 43-feature Input
class MulticlassDNN_RS43(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = nn.Sequential(
            nn.Linear(43, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_target = self.classifier(feats_encoded)

        return class_target  # Shape: (batch_size, 4)

#############################################################
# Binary PQC - Random Layers 24-Gate Circuit
class BinaryPQC_RL24Gate(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = QuantumFeatureEmbeddingBatch()

        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded.float())

        return class_probs  # Shape: (batch_size, 1)

#############################################################
# Multiclass PQC - Random Layers 24-Gate Circuit
class MulticlassPQC_RL24Gate(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodes features from dataset
        self.feature_encoder = QuantumFeatureEmbeddingBatch()

        # Classifies based on encoded features
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, features):
        feats_encoded = self.feature_encoder(features)
        class_probs = self.classifier(feats_encoded.float())

        return class_probs  # Shape: (batch_size, 1)

#############################################################
