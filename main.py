# main.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# self defined block
import config
from data_loader import load_labels, EEGDataset, collate_fn, split_patients, build_augmented_samples
from model import create_eegnet
from train import train_one_epoch, validate, test, evaluate_detailed


def main():
    torch.manual_seed(config.SEED)
    #load label
    label_dict = load_labels(config.ANNOTATION_CSV)

    # split sessions
    train_keys, val_keys, test_keys = split_patients(
        label_dict,
        test_ratio=config.TEST_RATIO,
        val_ratio=config.VAL_RATIO,
        seed=config.SEED
    )
 
    # create datset
    train_dataset_original = EEGDataset(config.DATA_ROOT, label_dict,
                                         samples_per_session=config.SAMPLES_PER_SESSION,
                                         allowed_keys=train_keys)
    
    augmented_samples = build_augmented_samples(
        base_samples=train_dataset_original.samples,
        seizure_target=1000,      
        num_augmented=40000,       
        seed=config.SEED
    )
    
    train_dataset = EEGDataset(config.DATA_ROOT, label_dict,
                               samples_list=augmented_samples,
                               chunk_size=config.CHUNK_SIZE)
    val_dataset = EEGDataset(config.DATA_ROOT, label_dict,
                             samples_per_session=config.SAMPLES_PER_SESSION,
                             allowed_keys=val_keys,
                             chunk_size=config.CHUNK_SIZE)
    test_dataset = EEGDataset(config.DATA_ROOT, label_dict,
                              samples_per_session=config.SAMPLES_PER_SESSION,
                              allowed_keys=test_keys,
                              chunk_size=config.CHUNK_SIZE)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")


    if len(train_dataset) == 0:
        print("Training set is empty, exit")
        return

  
    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn, drop_last=False)

  
    # create model
    model = create_eegnet(
        chunk_size=config.CHUNK_SIZE,
        num_electrodes=config.NUM_ELECTRODES,
        F1=config.F1,
        F2=config.F2,
        D=config.D,
        num_classes=config.NUM_CLASSES,
        kernel_1=config.KERNEL_1,
        kernel_2=config.KERNEL_2,
        dropout=config.DROPOUT
    )
    model = model.to(config.DEVICE)

    print("model structure:", model)

  
    # loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    total_epochs = config.EPOCHS
    warmup_epochs = 5

    # warmup round: 0.01x --> 1x
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)

    # cosine scheduler to supress lr in later training
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=5e-5)
    
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    

    # 7. training
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # dynamic LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"current learning rate :{current_lr}")
        
        # early stop & saving model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("--> best model has been saved!")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

  
    # test best model
    model.load_state_dict(torch.load("best_model.pth"))
    test_acc = test(model, test_loader, config.DEVICE)
    print(f"\ntest acc: {test_acc:.2f}%")

    
    labels, preds, probs, patients, sessions = evaluate_detailed(model, test_loader, config.DEVICE)

    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    import numpy as np
    from collections import defaultdict

    # sample confusion matrix
    cm_sample = confusion_matrix(labels, preds)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm_sample, annot=True, fmt='d', cmap='Blues')
    plt.title('Sample-level Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    #sample ROC
    auc_sample = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Sample AUC = {auc_sample:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Sample-level ROC')
    plt.legend()
    plt.show()

    #session level
    session_true = {}
    session_probs = defaultdict(list)
    session_votes = defaultdict(list)

    for true, pred, prob, pat, sess in zip(labels, preds, probs, patients, sessions):
        key = (pat, sess)
        session_true[key] = true  
        session_probs[key].append(prob)
        session_votes[key].append(pred)

    session_labels = []
    session_pred_probs = []
    session_pred_majority = []

    for key in session_true.keys():
        true_label = session_true[key]
        avg_prob = np.mean(session_probs[key])
        majority = np.bincount(session_votes[key]).argmax()
        session_labels.append(true_label)
        session_pred_probs.append(avg_prob)
        session_pred_majority.append(majority)

    # session confusion matrix
    cm_session = confusion_matrix(session_labels, session_pred_majority)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm_session, annot=True, fmt='d', cmap='Greens')
    plt.title('Session-level Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # session ROC
    auc_session = roc_auc_score(session_labels, session_pred_probs)
    fpr_s, tpr_s, _ = roc_curve(session_labels, session_pred_probs)
    plt.figure()
    plt.plot(fpr_s, tpr_s, label=f'Session AUC = {auc_session:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Session-level ROC')
    plt.legend()
    plt.show()

    test_acc = test(model, test_loader, config.DEVICE)
    print(f"\nTest Accuracy (from original test function): {test_acc:.2f}%")
    
    # plotting training cureve
    sns.set_theme(style="darkgrid", palette='muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training and Validation Curves', fontsize=16, fontweight='bold')

    epochs_range = range(1, len(train_losses) + 1)

    ax1.plot(epochs_range, train_losses, 'o-', label='Training Loss', lw=2, markersize=4, color='blue')
    ax1.plot(epochs_range, val_losses, 's-', label='Validation Loss', lw=2, markersize=4, color='orange')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss over Epochs', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(epochs_range, train_accs, 'o-', label='Training Accuracy', lw=2, markersize=4, color='blue')
    ax2.plot(epochs_range, val_accs, 's-', label='Validation Accuracy', lw=2, markersize=4, color='orange')
    ax2.axhline(y=test_acc, color='green', linestyle='--', lw=2, label=f'Test Accuracy ({test_acc:.2f}%)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy over Epochs', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("\n training curves has been saved as training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()
