def InitializeModel(model, load_path = None, classifier = "Binary"):

    ############ Conditionals for all model types ############## 
    ### DNN Various Input ###
    if model == "DNN8":
        #Initialize architecture here
        if classifier = "Binary":
            model = BinaryDNN_PureClassical()
        if classifier = "Multiclass":
            model = MulticlassDNN_PureClassical()
        
    else if model == "DNN40":
        #Initialize architecture here
        if classifier = "Binary":
            model = BinaryDNN_RS40()
        if classifier = "Multiclass":
            model = MulticlassDNN_RS40()

    else if model == "DNN43":
        #Initialize architecture here
        if classifier = "Binary":
            model = BinaryDNN_RS43()
        if classifier = "Multiclass":
            model = MulticlassDNN_RS43()

    ### PQC Models ###
    else if model == "RandomLayer":
        #Initialize PQC
        from QuantumCircuits import InitializePQC
        InitializePQC(model)

        #Initialize architecture here
        if classifier = "Binary":
            model = 
        if classifier = "Multiclass":
            model = 

    else if model == "StronglyEntangled":
        #Initialize PQC
        from QuantumCircuits import InitializePQC
        InitializePQC(model)

        #Initialize architecture here
        if classifier = "Binary":
            model = BinaryPQC()
        if classifier = "Multiclass":
            model = MulticlassPQC()

    else:
        print("Not a valid model choice...")
        return

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
# Binary PQC - Variable Circuit
class BinaryPQC(nn.Module):
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
# Multiclass PQC - Variable Circuit
class MulticlassPQC(nn.Module):
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
