import pandas as pd
from recbole.quick_start import run_recbole
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
import glob, os

# -------------------------
# DATA FILES
# -------------------------
dataset_name = "mydata_presplit_ranking"
dataset_dir = f"dataset/{dataset_name}"

USER_FILE = f"{dataset_dir}/{dataset_name}.user"
ITEM_FILE = f"{dataset_dir}/{dataset_name}.item"
TRAIN_FILE = f"{dataset_dir}/{dataset_name}.train.inter"
VALID_FILE = f"{dataset_dir}/{dataset_name}.valid.inter"
TEST_FILE = f"{dataset_dir}/{dataset_name}.test.inter"

# -------------------------
# VERIFY RAW DATA SPLITS
# -------------------------
df_train = pd.read_csv(TRAIN_FILE, sep='\t')
df_valid = pd.read_csv(VALID_FILE, sep='\t')
df_test  = pd.read_csv(TEST_FILE, sep='\t')

print(f"Train: {len(df_train)} interactions")
print(f"Valid: {len(df_valid)} interactions")
print(f"Test:  {len(df_test)} interactions")

# -------------------------
# LOAD COLUMN TYPES
# -------------------------
def get_load_type(df, id_fields):
    load = {}
    for col in df.columns:
        base_col = col.split(':')[0]
        if base_col in id_fields:
            load[base_col] = 'token'
        elif base_col == 'label':
            load[base_col] = 'float'
        else:
            load[base_col] = 'float'
    return load

load_col_config = {
    "inter": get_load_type(df_train, ['user_id','item_id']),
    "user": get_load_type(pd.read_csv(USER_FILE, sep='\t', nrows=1), ['user_id']),
    "item": get_load_type(pd.read_csv(ITEM_FILE, sep='\t', nrows=1), ['item_id']),
}

# -------------------------
# DELETE OLD CHECKPOINTS
# -------------------------
for f in glob.glob(f"saved/*{dataset_name}*.pth"):
    os.remove(f)
print("✅ Old checkpoints removed, training will start from scratch.")

# -------------------------
# CONFIG DICT
# -------------------------
config_dict = {
    'data_path': 'dataset',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'LABEL_FIELD': 'label',
    'field_separator': '\t',
    'load_col': load_col_config,
    'benchmark_filename': ['train','valid','test'],

    # Model params
    'embedding_size': 16,
    'mlp_hidden_size': [128,128,128],
    'cross_layer_num': 3,
    'reg_weight': 0.0001,
    'dropout_prob': 0.2,

    # Training
    'epochs': 100,
    'train_batch_size': 256,
    'eval_batch_size': 512,
    'learning_rate': 0.001,
    'stopping_step': 5,
    'eval_step': 1,

    # Metrics
    'metrics': ['Recall', 'NDCG', 'Precision'],
    'topk': [5,10,20],
    'valid_metric': 'Recall@5',

    # Evaluation args
    'eval_args': {
        'split': 'preserved',  
        'filter_inter_by_user_or_item': False,
        'group_by': 'user',
        'order': 'RO',
        'mode': 'full'
    },

    # Cold user/item handling
    'ignore_cold_user': False,
    'ignore_cold_item': False,

    # Device & progress
    'device': 'cuda',
    'show_progress': True,

    # Force training from scratch
    'checkpoint_dir': 'saved',
    'checkpoint_file': None
}

# -------------------------
# CREATE DATASET
# -------------------------
model_name = 'DeepFM'  # <-- change this to 'DCN', 'DIN', etc.

# Create config with the correct model
config = Config(model=model_name, dataset=dataset_name, config_dict=config_dict)

# Create dataset & loaders
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

print(f"\n✅ Internal vs External length check:")
print(f"   TRAIN: internal={len(train_data.dataset)}, external={len(df_train)}")
print(f"   VALID: internal={len(valid_data.dataset)}, external={len(df_valid)}")
print(f"   TEST : internal={len(test_data.dataset)}, external={len(df_test)}")

# Train
print(f"\n🚀 STARTING {model_name} TRAINING\n")
run_recbole(model=model_name, dataset=dataset_name, config_dict=config_dict)