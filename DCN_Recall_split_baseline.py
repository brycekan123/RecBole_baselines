import pandas as pd
import os
from recbole.quick_start import run_recbole

# File paths
train_csv = 'split_data_k_or_more/train.csv'
val_csv = 'split_data_k_or_more/valid.csv'
test_csv = 'split_data_k_or_more/test.csv'
dataset_dir = 'dataset/mydata_presplit_ranking'
train_inter = f'{dataset_dir}/mydata_presplit_ranking.train.inter'
val_inter = f'{dataset_dir}/mydata_presplit_ranking.valid.inter'
test_inter = f'{dataset_dir}/mydata_presplit_ranking.test.inter'
base_inter = f'{dataset_dir}/mydata_presplit_ranking.inter'

print("="*80)
print("DCN TRAINING WITH RECALL@K EVALUATION")
print("="*80)

# Convert CSV to RecBole format
if not os.path.exists(dataset_dir):
    print("\nConverting pre-split CSVs to RecBole format...")
    
    # Load CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"Train: {train_df.shape}, Labels: {train_df['label'].value_counts().to_dict()}")
    print(f"Val: {val_df.shape}, Labels: {val_df['label'].value_counts().to_dict()}")
    print(f"Test: {test_df.shape}, Labels: {test_df['label'].value_counts().to_dict()}")
    
    # Create column mapping
    column_mapping = {
        'user_id': 'user_id:token',
        'job_id': 'item_id:token',
        'label': 'label:float',
    }
    
    feature_cols = [col for col in train_df.columns if col not in ['user_id', 'job_id', 'label']]
    for col in feature_cols:
        column_mapping[col] = f'{col}:float'
    
    print(f"\nMapping {len(column_mapping)} columns (including {len(feature_cols)} features)")
    
    # Convert
    train_inter_df = train_df.rename(columns=column_mapping)
    val_inter_df = val_df.rename(columns=column_mapping)
    test_inter_df = test_df.rename(columns=column_mapping)
    
    # Save
    os.makedirs(dataset_dir, exist_ok=True)
    train_inter_df.to_csv(train_inter, sep='\t', index=False)
    val_inter_df.to_csv(val_inter, sep='\t', index=False)
    test_inter_df.to_csv(test_inter, sep='\t', index=False)
    train_inter_df.to_csv(base_inter, sep='\t', index=False)
    
    print("✅ Created all .inter files\n")
else:
    print("✅ Using existing .inter files\n")

# Read columns
df_temp = pd.read_csv(train_inter, sep='\t', nrows=1)
all_columns = [col.split(':')[0] for col in df_temp.columns]

print(f"Total columns: {len(all_columns)}")
print(f"Features: {len([c for c in all_columns if c not in ['user_id', 'item_id', 'label']])}")
print(f"'label' in columns: {'label' in all_columns}\n")

print("="*80)
print("TRAINING CONFIGURATION")
print("="*80)

config_dict = {
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'LABEL_FIELD': 'label',
    'field_separator': '\t',
    
    'load_col': {'inter': all_columns},
    'numerical_features': [col for col in all_columns if col not in ['user_id', 'item_id', 'label']],
    
    # Model hyperparameters
    'embedding_size': 16,
    'mlp_hidden_size': [128, 128, 128],
    'cross_layer_num': 3,
    'reg_weight': 0.0001,
    'dropout_prob': 0.2,
    
    # Training settings
    'epochs': 100,
    'train_batch_size': 2048,
    'eval_batch_size': 4096,
    'learning_rate': 0.001,
    'stopping_step': 10,
    'eval_step': 1,
    
    # Recall@K metrics
    'metrics': ['Recall', 'NDCG', 'Precision'],
    'topk': [5, 10, 20],
    'valid_metric': 'Recall@5',
    
    # Evaluation settings
    'eval_args': {
        'split': {'LS': 'valid_and_test'},
        'group_by': 'user',
        'order': 'RO',
        'mode': 'full',  # Ranks all items for each user
    },
    
    # No negative sampling (use your labeled data as-is)
    'train_neg_sample_args': None,
    
    'device': 'cuda',
    'show_progress': True,
}

print(f"Model: DCN")
print(f"Metrics: Recall@5, Recall@10, Recall@20, NDCG, Precision")
print(f"Valid metric: Recall@5")
print(f"Evaluation mode: full (ranks all items)")
print(f"\n⚠️  NOTE:")
print(f"  - Training: Uses your labeled data (with pre-sampled negatives)")
print(f"  - Evaluation: Ranks ALL items in dataset for each user")
print(f"  - This gives valid Recall@K, but not pure '1+99' evaluation")
print("="*80 + "\n")

# Train model
run_recbole(model='DCN', dataset='mydata_presplit_ranking', config_dict=config_dict)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print("Check the output above for:")
print("  - Training loss (should decrease)")
print("  - Validation Recall@5 (should be > 0)")
print("  - Test Recall@5, Recall@10, Recall@20")
print("  - Model saved in 'saved/' directory")
print("="*80)