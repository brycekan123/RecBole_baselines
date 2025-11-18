import pandas as pd
import os
from recbole.quick_start import run_recbole

# File paths
train_csv = 'split_data_k_or_more/train.csv'
val_csv = 'split_data_k_or_more/valid.csv'
test_csv = 'split_data_k_or_more/test.csv'
dataset_dir = 'dataset/mydata_presplit'
train_inter = f'{dataset_dir}/mydata_presplit.train.inter'
val_inter = f'{dataset_dir}/mydata_presplit.valid.inter'
test_inter = f'{dataset_dir}/mydata_presplit.test.inter'
base_inter = f'{dataset_dir}/mydata_presplit.inter'

# Check if ALL files exist (including base)
if not (os.path.exists(train_inter) and os.path.exists(val_inter) and os.path.exists(test_inter) and os.path.exists(base_inter)):
    print("="*80)
    print("Converting pre-split CSVs to RecBole format (first time only)")
    print("="*80)
    
    # Load your 3 pre-split CSVs
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
    
    # Convert all 3 splits
    train_inter_df = train_df.rename(columns=column_mapping)
    val_inter_df = val_df.rename(columns=column_mapping)
    test_inter_df = test_df.rename(columns=column_mapping)
    
    # Save
    os.makedirs(dataset_dir, exist_ok=True)
    train_inter_df.to_csv(train_inter, sep='\t', index=False)
    val_inter_df.to_csv(val_inter, sep='\t', index=False)
    test_inter_df.to_csv(test_inter, sep='\t', index=False)
    train_inter_df.to_csv(base_inter, sep='\t', index=False)
    
    print("\n✅ Created all .inter files")
else:
    print("✅ Using existing .inter files")

# DIAGNOSTIC: Check what's in the files
print("\n" + "="*80)
print("DIAGNOSTIC: Checking .inter files")
print("="*80)

train_check = pd.read_csv(train_inter, sep='\t')
print(f"\n1. Train .inter file:")
print(f"   Shape: {train_check.shape}")
print(f"   Columns: {train_check.columns[:5].tolist()} ... (showing first 5)")
print(f"   'label:float' exists: {'label:float' in train_check.columns}")
if 'label:float' in train_check.columns:
    print(f"   Label distribution: {train_check['label:float'].value_counts().to_dict()}")
    print(f"   Sample rows:")
    print(train_check[['user_id:token', 'item_id:token', 'label:float']].head(3))

# Read columns
df_temp = pd.read_csv(train_inter, sep='\t', nrows=1)
all_columns = [col.split(':')[0] for col in df_temp.columns]

print(f"\n2. Parsed columns for RecBole:")
print(f"   Total: {len(all_columns)}")
print(f"   First 5: {all_columns[:5]}")
print(f"   'label' in columns: {'label' in all_columns}")

print("\n" + "="*80)
print("Training DCN with pre-split data")
print("="*80)

config_dict = {
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'LABEL_FIELD': 'label',
    'field_separator': '\t',
    
    'load_col': {'inter': all_columns},
    'numerical_features': [col for col in all_columns if col not in ['user_id', 'item_id', 'label']],
    
    # Model
    'embedding_size': 16,
    'mlp_hidden_size': [128, 128, 128],
    'cross_layer_num': 3,
    'reg_weight': 0.0001,
    'dropout_prob': 0.2,
    
    # Training
    'epochs': 100,
    'train_batch_size': 2048,
    'eval_batch_size': 4096,
    'learning_rate': 0.001,
    'stopping_step': 10,
    'eval_step': 1,
    
    # Metrics
    'metrics': ['AUC', 'LogLoss'],
    'valid_metric': 'AUC',
    
    # Use pre-split files
    'eval_args': {
        'split': {'LS': 'valid_and_test'},
        'mode': 'labeled',
    },
    
    'device': 'cuda',
    'show_progress': True,
}

print(f"\n3. Config check:")
print(f"   LABEL_FIELD: {config_dict['LABEL_FIELD']}")
print(f"   load_col includes 'label': {'label' in config_dict['load_col']['inter']}")
print(f"   Number of numerical features: {len(config_dict['numerical_features'])}")
print(f"   'label' in numerical_features: {'label' in config_dict['numerical_features']}")

print("\n" + "="*80)
print("Starting RecBole training...")
print("="*80 + "\n")

# Add a simple baseline check before training
print("🔍 SANITY CHECK: Computing baseline label distribution AUC")
val_check = pd.read_csv(val_inter, sep='\t')
if 'label:float' in val_check.columns:
    pos_ratio = (val_check['label:float'] == 1).mean()
    print(f"   Validation set: {pos_ratio*100:.2f}% positive")
    print(f"   Expected AUC if predicting always positive: ~{pos_ratio:.3f}")
    print(f"   Expected AUC if predicting always negative: ~{1-pos_ratio:.3f}")
    print(f"   Random baseline AUC: ~0.5")
    print(f"   Your model should be ABOVE 0.5 if learning anything!\n")

run_recbole(model='DCN', dataset='mydata_presplit', config_dict=config_dict)