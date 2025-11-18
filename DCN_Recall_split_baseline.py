import pandas as pd
from recbole.quick_start import run_recbole

dataset_name = "mydata_presplit_ranking"
dataset_dir = "dataset/mydata_presplit_user_ranking"

USER_FILE = f"{dataset_dir}/{dataset_name}.user"
ITEM_FILE = f"{dataset_dir}/{dataset_name}.item"
TRAIN_FILE = f"{dataset_dir}/{dataset_name}.train"
VALID_FILE = f"{dataset_dir}/{dataset_name}.valid"
TEST_FILE = f"{dataset_dir}/{dataset_name}.test"

# -------------------------
# VERIFICATION: Check data split
# -------------------------
print("="*60)
print("VERIFYING DATA SPLIT")
print("="*60)

# Load full datasets
df_train = pd.read_csv(TRAIN_FILE, sep='\t')
df_valid = pd.read_csv(VALID_FILE, sep='\t')
df_test = pd.read_csv(TEST_FILE, sep='\t')

# Extract user_id column (handle :token suffix)
train_user_col = [col for col in df_train.columns if 'user_id' in col][0]
valid_user_col = [col for col in df_valid.columns if 'user_id' in col][0]
test_user_col = [col for col in df_test.columns if 'user_id' in col][0]

train_users = set(df_train[train_user_col].unique())
valid_users = set(df_valid[valid_user_col].unique())
test_users = set(df_test[test_user_col].unique())

print(f"\n📊 Dataset Statistics:")
print(f"   Train: {len(df_train)} interactions, {len(train_users)} unique users")
print(f"   Valid: {len(df_valid)} interactions, {len(valid_users)} unique users")
print(f"   Test:  {len(df_test)} interactions, {len(test_users)} unique users")

# Check overlaps
users_in_all = train_users & valid_users & test_users
users_only_train = train_users - valid_users - test_users
users_only_valid = valid_users - train_users - test_users
users_only_test = test_users - train_users - valid_users
users_missing_from_train = (valid_users | test_users) - train_users
users_missing_from_valid = (train_users | test_users) - valid_users
users_missing_from_test = (train_users | valid_users) - test_users

print(f"\n✅ User Coverage:")
print(f"   Users in ALL three splits: {len(users_in_all)}")
print(f"   Users only in train: {len(users_only_train)}")
print(f"   Users only in valid: {len(users_only_valid)}")
print(f"   Users only in test: {len(users_only_test)}")

if len(users_missing_from_train) > 0:
    print(f"\n⚠️  WARNING: {len(users_missing_from_train)} users in valid/test but NOT in train!")
    print(f"   Sample users: {list(users_missing_from_train)[:5]}")

if len(users_missing_from_valid) > 0:
    print(f"\n⚠️  WARNING: {len(users_missing_from_valid)} users in train/test but NOT in valid!")
    print(f"   Sample users: {list(users_missing_from_valid)[:5]}")

if len(users_missing_from_test) > 0:
    print(f"\n⚠️  WARNING: {len(users_missing_from_test)} users in train/valid but NOT in test!")
    print(f"   Sample users: {list(users_missing_from_test)[:5]}")

# Final verification
all_users = train_users | valid_users | test_users
coverage_percentage = (len(users_in_all) / len(all_users)) * 100

print(f"\n📈 Coverage Report:")
print(f"   Total unique users across all splits: {len(all_users)}")
print(f"   Users appearing in all splits: {len(users_in_all)} ({coverage_percentage:.1f}%)")

if coverage_percentage == 100.0:
    print(f"\n✅ PERFECT! All users appear in train, valid, AND test splits!")
elif coverage_percentage >= 95.0:
    print(f"\n✅ GOOD! {coverage_percentage:.1f}% of users appear in all splits.")
else:
    print(f"\n❌ PROBLEM! Only {coverage_percentage:.1f}% of users appear in all splits.")
    response = input("\nContinue training anyway? (yes/no): ")
    if response.lower() != 'yes':
        print("Exiting. Please fix data split first.")
        exit()

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

# -------------------------
# Load columns for RecBole
# -------------------------
df_train_sample = pd.read_csv(TRAIN_FILE, sep='\t', nrows=1)
inter_columns = [col.split(':')[0] for col in df_train_sample.columns]
inter_load = {}
for col in inter_columns:
    if col in ['user_id', 'item_id']:
        inter_load[col] = 'token'
    elif col == 'label':
        inter_load[col] = 'float'
    else:
        inter_load[col] = 'float'

df_user = pd.read_csv(USER_FILE, sep='\t', nrows=1)
user_columns = [col.split(':')[0] for col in df_user.columns]
user_load = {col: 'token' if col == 'user_id' else 'float' for col in user_columns}

df_item = pd.read_csv(ITEM_FILE, sep='\t', nrows=1)
item_columns = [col.split(':')[0] for col in df_item.columns]
item_load = {col: 'token' if col == 'item_id' else 'float' for col in item_columns}

load_col_config = {
    "inter": inter_load,
    "user": user_load,
    "item": item_load
}

# -------------------------
# RecBole config
# -------------------------
config_dict = {
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'LABEL_FIELD': 'label',
    'field_separator': '\t',
    'load_col': load_col_config,
    
    # Model params
    'embedding_size': 16,
    'mlp_hidden_size': [128,128,128],
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
    'metrics': ['Recall', 'NDCG', 'Precision'],
    'topk': [5,10,20],
    'valid_metric': 'Recall@5',
    
    # Evaluation args for pre-split data
    'eval_args': {
        'split': {'RS': [0.8, 0.1, 0.1]},
        'group_by': 'user',
        'order': 'RO',
        'mode': 'full'
    },
    
    'device': 'cuda',
    'show_progress': True,
}

# -------------------------
# Run DCN
# -------------------------
run_recbole(model='DCN', dataset=dataset_name, config_dict=config_dict)