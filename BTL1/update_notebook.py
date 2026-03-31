import json

# Read the notebook
with open('multimodal.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New cell 1: Updated few-shot training functions (replaces cell 66)
new_cell_66_source = '''def extract_features(df, split_name, desc):
    features_list = []
    batch_size = 32
    for i in tqdm(range(0, len(df), batch_size), desc=desc):
        batch_idx = df.iloc[i : i + batch_size]["idx"].tolist()
        batch_imgs = [flickr8k[split_name][idx]["image"].convert("RGB") for idx in batch_idx]
        img_tensors = torch.stack([clip_preprocess(img) for img in batch_imgs]).to(DEVICE)
        with torch.no_grad():
            f = clip_model.encode_image(img_tensors)
            f /= f.norm(dim=-1, keepdim=True)
            features_list.append(f.cpu().numpy())
    return np.concatenate(features_list, axis=0)


class FocalLossModel(nn.Module):
    """PyTorch model with focal loss for imbalanced classification"""
    def __init__(self, input_dim, num_classes, alpha=None, gamma=2.0):
        super(FocalLossModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, x):
        return self.linear(x)
    
    def focal_loss(self, logits, targets):
        """Focal Loss: L = -alpha * (1 - pt)^gamma * log(pt)"""
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


def train_few_shot_variants(df_train=df_train, df_val=df_validate):
    """Trains 3 few-shot models: unweighted LogLoss, weighted LogLoss, weighted focal loss"""
    from sklearn.linear_model import SGDClassifier
    from sklearn.utils.class_weight import compute_class_weight

    print("="*70)
    print("EXTRACTING CLIP FEATURES (SHARED ACROSS ALL MODELS)")
    print("="*70)
    X_train = extract_features(df_train, "train", "Encoding Train Images")
    y_train = np.array(df_train["Assigned_Label"].tolist())
    X_val = extract_features(df_val, "validation", "Encoding Val Images")
    y_val = np.array(df_val["Assigned_Label"].tolist())
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    results = {}
    
    # MODEL 1: Unweighted, Log Loss
    print("\\n" + "="*70)
    print("MODEL 1: UNWEIGHTED, LOG LOSS (BASELINE)")
    print("="*70)
    model_unweighted = SGDClassifier(loss='log_loss', learning_rate='optimal', early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=5, random_state=SEED, class_weight=None, verbose=0)
    model_unweighted.fit(X_train, y_train)
    val_preds_m1 = model_unweighted.predict(X_val)
    val_acc_m1 = accuracy_score(y_val, val_preds_m1)
    print(f"✓ Model 1 trained. Validation Accuracy: {val_acc_m1:.4f}")
    results['unweighted_logloss'] = {'model': model_unweighted, 'val_accuracy': val_acc_m1, 'name': 'Unweighted LogLoss'}
    
    # MODEL 2: Weighted (Balanced), Log Loss
    print("\\n" + "="*70)
    print("MODEL 2: WEIGHTED (BALANCED), LOG LOSS")
    print("="*70)
    model_weighted_logloss = SGDClassifier(loss='log_loss', learning_rate='optimal', early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=5, random_state=SEED, class_weight='balanced', verbose=0)
    model_weighted_logloss.fit(X_train, y_train)
    val_preds_m2 = model_weighted_logloss.predict(X_val)
    val_acc_m2 = accuracy_score(y_val, val_preds_m2)
    print(f"✓ Model 2 trained. Validation Accuracy: {val_acc_m2:.4f}")
    results['weighted_logloss'] = {'model': model_weighted_logloss, 'val_accuracy': val_acc_m2, 'name': 'Weighted LogLoss'}
    
    # MODEL 3: Weighted (Balanced), Focal Loss (PyTorch)
    print("\\n" + "="*70)
    print("MODEL 3: WEIGHTED (BALANCED), FOCAL LOSS (PyTorch)")
    print("="*70)
    X_train_torch = torch.from_numpy(X_train).float().to(DEVICE)
    y_train_torch = torch.from_numpy(pd.factorize(y_train)[0]).long().to(DEVICE)
    X_val_torch = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_torch = torch.from_numpy(pd.factorize(y_val)[0]).long().to(DEVICE)
    unique_classes = sorted(np.unique(y_train))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    num_classes = len(unique_classes)
    alpha_weights = torch.tensor([class_weights[cls] for cls in unique_classes], dtype=torch.float32, device=DEVICE)
    focal_model = FocalLossModel(input_dim=X_train.shape[1], num_classes=num_classes, alpha=alpha_weights, gamma=2.0).to(DEVICE)
    optimizer = torch.optim.Adam(focal_model.parameters(), lr=0.001)
    best_val_acc, patience, no_improve_count = 0, 5, 0
    print(f"Training for up to 100 epochs with early stopping (patience={patience})...")
    for epoch in range(100):
        focal_model.train()
        optimizer.zero_grad()
        logits = focal_model(X_train_torch)
        loss = focal_model.focal_loss(logits, y_train_torch)
        loss.backward()
        optimizer.step()
        focal_model.eval()
        with torch.no_grad():
            val_logits = focal_model(X_val_torch)
            val_preds_m3 = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_acc_m3 = accuracy_score(y_val_torch.cpu().numpy(), val_preds_m3)
        if val_acc_m3 > best_val_acc:
            best_val_acc, no_improve_count = val_acc_m3, 0
        else:
            no_improve_count += 1
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Train Loss={loss.item():.4f}, Val Acc={val_acc_m3:.4f}")
        if no_improve_count >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    print(f"✓ Model 3 trained. Final Validation Accuracy: {best_val_acc:.4f}")
    results['weighted_focal'] = {'model': focal_model, 'val_accuracy': best_val_acc, 'name': 'Weighted Focal Loss',
        'class_to_idx': class_to_idx, 'unique_classes': unique_classes}
    print("\\n" + "="*70 + "\\nTRAINING SUMMARY\\n" + "="*70)
    for key, result in results.items():
        print(f"{result['name']:30s} | Val Acc: {result['val_accuracy']:.4f}")
    return results'''

# Split source into lines
nb['cells'][66]['source'] = [line + '\n' for line in new_cell_66_source.split('\n')[:-1]] + [new_cell_66_source.split('\n')[-1]]

# Update cell 67 (the training call) - few_shot_model = train_few_shot()
new_cell_67_source = '''model_results = train_few_shot_variants()
model_m1 = model_results['unweighted_logloss']['model']
model_m2 = model_results['weighted_logloss']['model']
model_m3 = model_results['weighted_focal']['model']
class_to_idx_m3 = model_results['weighted_focal']['class_to_idx']
unique_classes_m3 = model_results['weighted_focal']['unique_classes']'''

nb['cells'][67]['source'] = [line + '\n' for line in new_cell_67_source.split('\n')[:-1]] + [new_cell_67_source.split('\n')[-1]]

# Update cell 68 (predictions)
new_cell_68_source = '''# Extract test features once and reuse for all models
X_test_features = extract_features(df_test, "test", "Encoding Test Images")

# MODEL 1: Unweighted LogLoss Predictions
predictions_m1 = model_m1.predict(X_test_features)
predictions_m1 = [" ".join(x.split(" ")[:3]) + "..." for x in predictions_m1]

# MODEL 2: Weighted LogLoss Predictions
predictions_m2 = model_m2.predict(X_test_features)
predictions_m2 = [" ".join(x.split(" ")[:3]) + "..." for x in predictions_m2]

# MODEL 3: Weighted Focal Loss Predictions
X_test_torch = torch.from_numpy(X_test_features).float().to(DEVICE)
model_m3.eval()
with torch.no_grad():
    logits_m3 = model_m3(X_test_torch)
    preds_idx_m3 = torch.argmax(logits_m3, dim=1).cpu().numpy()
    # Map indices back to class labels
    idx_to_class = {v: k for k, v in class_to_idx_m3.items()}
    predictions_m3 = [idx_to_class[idx] for idx in preds_idx_m3]
    predictions_m3 = [" ".join(x.split(" ")[:3]) + "..." for x in predictions_m3]

print(f"✓ Predictions generated for all 3 models on {len(X_test_features)} test samples")'''

nb['cells'][68]['source'] = [line + '\n' for line in new_cell_68_source.split('\n')[:-1]] + [new_cell_68_source.split('\n')[-1]]

# Update cell 69 (evaluation) - comprehensive
new_cell_69_source = '''ground_truth = [" ".join(x.split(" ")[:3]) + "..." for x in df_test["Assigned_Label"].tolist()]

# Helper function to evaluate a model
def evaluate_model(predictions, ground_truth, model_name):
    """Evaluate and display metrics for a single model"""
    accuracy = accuracy_score(predictions, ground_truth)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='weighted'
    )
    
    # Get per-class F1 for macro-F1 calculation
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        ground_truth, predictions, average=None
    )
    macro_f1 = f1_per_class.mean()
    
    # Get per-class recall for macro-recall
    _, recall_per_class, _, _ = precision_recall_fscore_support(
        ground_truth, predictions, average=None
    )
    macro_recall = recall_per_class.mean()
    
    print(f"\\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    print(f"Accuracy (weighted):  {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1-Score (weighted):  {f1:.4f}")
    print(f"F1-Score (macro):     {macro_f1:.4f}")
    print(f"Recall (macro):       {macro_recall:.4f}")
    print()
    print(classification_report(ground_truth, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=shortened_classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/confusion_matrix_{model_name.replace(' ', '_').lower()}.png", 
                dpi=150, bbox_inches="tight")
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_weighted': f1,
        'f1_macro': macro_f1,
        'recall_macro': macro_recall
    }

# Evaluate all three models
print("\\n" + "="*70)
print("EVALUATING ALL 3 MODELS ON TEST SET")
print("="*70)

results_m1 = evaluate_model(predictions_m1, ground_truth, "Model 1: Unweighted LogLoss")
results_m2 = evaluate_model(predictions_m2, ground_truth, "Model 2: Weighted LogLoss")
results_m3 = evaluate_model(predictions_m3, ground_truth, "Model 3: Weighted Focal Loss")

# Create comparison summary table
print("\\n" + "="*70)
print("SUMMARY COMPARISON TABLE")
print("="*70)
comparison_df = pd.DataFrame({
    'Model': ['Unweighted LogLoss', 'Weighted LogLoss', 'Weighted Focal Loss'],
    'Accuracy': [results_m1['accuracy'], results_m2['accuracy'], results_m3['accuracy']],
    'Precision (W)': [results_m1['precision'], results_m2['precision'], results_m3['precision']],
    'Recall (W)': [results_m1['recall'], results_m2['recall'], results_m3['recall']],
    'F1 (W)': [results_m1['f1_weighted'], results_m2['f1_weighted'], results_m3['f1_weighted']],
    'F1 (M)': [results_m1['f1_macro'], results_m2['f1_macro'], results_m3['f1_macro']],
    'Recall (M)': [results_m1['recall_macro'], results_m2['recall_macro'], results_m3['recall_macro']]
})
comparison_df = comparison_df.round(4)
display(comparison_df)

# Visualization: Compare key metrics
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
metrics = ['Accuracy', 'Precision (W)', 'Recall (W)', 'F1 (W)', 'F1 (M)', 'Recall (M)']
model_names = ['Unweighted', 'Weighted LogLoss', 'Weighted Focal']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
    values = [
        results_m1[metric.lower().replace(' (w)', '_weighted').replace(' (m)', '_macro').replace(' ', '_')],
        results_m2[metric.lower().replace(' (w)', '_weighted').replace(' (m)', '_macro').replace(' ', '_')],
        results_m3[metric.lower().replace(' (w)', '_weighted').replace(' (m)', '_macro').replace(' ', '_')]
    ]
    bars = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_title(metric, fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.tick_params(axis='x', rotation=45)

plt.suptitle("Few-Shot Classification: Model Comparison", fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig("outputs/figures/few_shot_models_comparison.png", dpi=150, bbox_inches="tight")
plt.show()'''

nb['cells'][69]['source'] = [line + '\n' for line in new_cell_69_source.split('\n')[:-1]] + [new_cell_69_source.split('\n')[-1]]

# Write back
with open('multimodal.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("✓ Notebook updated successfully!")
print("Cell 66: New train_few_shot_variants() + FocalLossModel class")
print("Cell 67: Training call to variants")
print("Cell 68: Predictions for all 3 models")
print("Cell 69: Comprehensive evaluation for all 3 models")
