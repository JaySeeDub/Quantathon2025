#!/usr/bin/env python
# coding: utf-8

def train(model, n_epochs, lr, )
    for epoch in range(n_epochs):
        model.train()
        train_loss, train_metric = 0.0, 0.0
    
        for features, target in train_loader:
            features, target = features.to(device), target.unsqueeze(-1).to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = bce(outputs, target)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            train_metric += metric(outputs, target)
    
        train_loss /= len(train_loader)
        train_metric /= len(train_loader)
    
        # Validation
        model.eval()
        val_loss, val_metric = 0.0, 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.unsqueeze(-1).to(device)
                outputs = model(X_val)
                val_loss += bce(outputs, y_val).item()
                val_metric += metric(outputs, y_val)
    
        val_loss /= len(val_loader)
        val_metric /= len(val_loader)
    
        # Logging
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_metric'].append(train_metric)
        history['val_loss'].append(val_loss)
        history['val_metric'].append(val_metric)
    
        # Report every 10th epoch
        if epoch % 10 == 9:
            print(f'Epoch [{epoch+1}/{n_epochs}] | Train Loss: {train_loss:.4f} | '
            f'Train Acc: {train_metric:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metric:.4f}')
    
        # Save best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            os.makedirs("models", exist_ok=True)
            save_path = "models/best_accuracy_epoch{epoch}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "history": history,
            }, save_path)
            print(f'Epoch [{epoch+1}] had best val_metric')
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            save_path = f"models/best_valLoss_epoch{epoch}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "history": history,
            }, save_path)
            print(f'Epoch [{epoch+1}] had best val_Loss')



def save(model, save_path=None)
    ## Save Model
    # Create output directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    if save_path == None:
        # Timestamp for unique filenames
        timestamp = datetime.now().strftime("%m%d_%H%M")
        save_path = f"models/final_model_{n_epochs}epochs_{timestamp}.pt"
    
    # Save model states and tracked data in a single file
    torch.save({
        "state_dict": model.state_dict(),
        "losses": losses,
    }, save_path)
    
    print(f"Model and statistics saved to {save_path}")



def eval(model, test_loader)
    # --- Set model to evaluation mode ---
    model.eval()
    
    all_targets = []
    all_preds = []
    all_outputs = []
    
    with torch.no_grad():
        for features, target in test_loader:
            features = features.to(device)
            target = target.to(device).float().unsqueeze(-1)
    
            outputs = model(features)
            preds = (outputs > 0.5).float()
    
            all_targets.append(target.cpu())
            all_preds.append(preds.cpu())
            all_outputs.append(outputs.cpu())

    # --- Concatenate all batches ---
    all_targets = torch.cat(all_targets).squeeze().long().numpy()  # integers 0/1
    all_preds = torch.cat(all_preds).squeeze().long().numpy()
    all_outputs = torch.cat(all_outputs).squeeze().numpy()             # floats in [0,1]
    
    # --- Sanity check ---
    print(all_targets.shape, all_preds.shape, all_outputs.shape)
    print(np.unique(all_targets))  # should be [0,1]
    
    # --- Compute Metrics ---
    cm = confusion_matrix(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_outputs)  # should work now
    f1 = f1_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    
    # Critical Success Index (CSI)
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    csi = tp / (tp + fn + fp)
    
    print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}, CSI: {csi:.4f}")
    
    # --- Plot Confusion Matrix ---
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # --- Plot ROC Curve ---
    fpr, tpr, thresholds = roc_curve(all_targets, all_outputs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color='blue')
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    
    # --- Plot Metrics Bar Chart ---
    metrics = {'Accuracy': acc, 'F1 Score': f1, 'CSI': csi}
    plt.figure(figsize=(6,4))
    plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange', 'green'])
    plt.ylim(0,1)
    plt.title('Classification Metrics')
    plt.show()