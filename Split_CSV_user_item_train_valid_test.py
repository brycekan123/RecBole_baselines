import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit

# -------------------------
# File paths
# -------------------------
CSV_FILE = "label_pref_no_text_onehotenc_k_or_more.csv"
DATASET_DIR = "dataset/mydata_presplit_user_ranking"

os.makedirs(DATASET_DIR, exist_ok=True)

USER_FILE = os.path.join(DATASET_DIR, "mydata_presplit_ranking.user")
ITEM_FILE = os.path.join(DATASET_DIR, "mydata_presplit_ranking.item")
TRAIN_FILE = os.path.join(DATASET_DIR, "mydata_presplit_ranking.train")
VALID_FILE = os.path.join(DATASET_DIR, "mydata_presplit_ranking.valid")
TEST_FILE = os.path.join(DATASET_DIR, "mydata_presplit_ranking.test")

# -------------------------
# Configuration
# -------------------------
VALID_RATIO = 0.1  # 10% of each user's interactions for validation
TEST_RATIO = 0.2   # 10% of each user's interactions for test
RANDOM_STATE = 42

# -------------------------
# Load CSV
# -------------------------
print("Loading CSV...")
df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} rows")
print(f"Label distribution:\n{df['label'].value_counts()}")

# -------------------------
# Define column groups
# -------------------------
user_keys = ["user_id"]
item_keys = ["job_id"]

user_prefixes = [
    "masters_major_", "second_major_", "undergrad_major_",
    "in_person_commitment_", "academic_level_",
    "interest_", "skill_",
]
user_boolean = ["commit_to_summer"]
user_numeric = ["gpa", "months_experience", "undergrad_gpa",
                "hours_per_week", "length_of_commitment", "num_publications"]

item_prefixes = [
    "industry_", "job_in_person_commitment_", "min_academic_level_",
    "required_skill_", "acceptable_major_"
]
item_numeric = ["min_gpa", "min_months_experience", "min_hours_per_week",
                "min_length_of_commitment", "min_num_publications"]

# -------------------------
# Classify columns
# -------------------------
user_cols = []
item_cols = []
unassigned = []

for col in df.columns:
    if col in user_keys + user_boolean + user_numeric:
        user_cols.append(col)
    elif any(col.startswith(p) for p in user_prefixes):
        user_cols.append(col)
    elif col in item_keys + item_numeric:
        item_cols.append(col)
    elif any(col.startswith(p) for p in item_prefixes):
        item_cols.append(col)
    elif col in ["row_id"]:
        continue
    elif col == "label":
        continue
    else:
        unassigned.append(col)

# -------------------------
# Verification
# -------------------------
if unassigned:
    raise ValueError(f"Unassigned columns detected: {unassigned}")

overlap = set(user_cols).intersection(item_cols)
if overlap:
    raise ValueError(f"Overlapping columns: {overlap}")

print("✅ All columns classified successfully.")
print(f"   User features: {len(user_cols)}")
print(f"   Item features: {len(item_cols)}")

# -------------------------
# Save .user
# -------------------------
print("\nPreparing .user file...")
user_feature_cols = [col for col in user_cols if col != "user_id"]
df_users = df[["user_id"] + user_feature_cols].drop_duplicates("user_id")

user_file_cols = ["user_id:token"] + [f"{col}:float" for col in user_feature_cols]
df_users_renamed = df_users.copy()
df_users_renamed.columns = user_file_cols

df_users_renamed.to_csv(USER_FILE, sep='\t', index=False)
print(f"✅ Saved {USER_FILE}")
print(f"   Shape: {df_users_renamed.shape}")
print(f"   Unique users: {len(df_users)}")

# -------------------------
# Save .item (rename job_id to item_id)
# -------------------------
print("\nPreparing .item file...")
item_feature_cols = [col for col in item_cols if col != "job_id"]
df_items = df[["job_id"] + item_feature_cols].drop_duplicates("job_id")

df_items = df_items.rename(columns={"job_id": "item_id"})

item_file_cols = ["item_id:token"] + [f"{col}:float" for col in item_feature_cols]
df_items_renamed = df_items.copy()
df_items_renamed.columns = item_file_cols

df_items_renamed.to_csv(ITEM_FILE, sep='\t', index=False)
print(f"✅ Saved {ITEM_FILE}")
print(f"   Shape: {df_items_renamed.shape}")
print(f"   Unique items: {len(df_items)}")

# -------------------------
# Split interactions per user into train/valid/test
# -------------------------
print("\nSplitting interactions...")
df_inter = df[df['label'] == 1][['user_id', 'job_id', 'label']].copy()
df_inter = df_inter.rename(columns={"job_id": "item_id"})

# Check interactions per user
interactions_per_user = df_inter.groupby('user_id').size()
print(f"   Total positive interactions: {len(df_inter)}")
print(f"   Users with interactions: {len(interactions_per_user)}")
print(f"   Min interactions per user: {interactions_per_user.min()}")
print(f"   Max interactions per user: {interactions_per_user.max()}")
print(f"   Mean interactions per user: {interactions_per_user.mean():.2f}")

# Filter users with at least 3 interactions (needed for train/valid/test split)
users_with_enough_data = interactions_per_user[interactions_per_user >= 3].index
df_inter_filtered = df_inter[df_inter['user_id'].isin(users_with_enough_data)]

if len(df_inter_filtered) < len(df_inter):
    print(f"   ⚠️  Filtered out {len(df_inter) - len(df_inter_filtered)} interactions from users with <3 interactions")
    df_inter = df_inter_filtered

# Split per user to ensure every user is in all splits
train_list = []
valid_list = []
test_list = []

for user_id, user_data in df_inter.groupby('user_id'):
    user_data = user_data.sample(frac=1, random_state=RANDOM_STATE)  # Shuffle
    n = len(user_data)
    
    n_valid = max(1, int(n * VALID_RATIO))
    n_test = max(1, int(n * TEST_RATIO))
    n_train = n - n_valid - n_test
    
    if n_train < 1:
        # If not enough data, put at least 1 in each
        n_train = max(1, n - 2)
        n_valid = 1
        n_test = 1 if n > 2 else 0
    
    train_list.append(user_data.iloc[:n_train])
    valid_list.append(user_data.iloc[n_train:n_train+n_valid])
    if n_test > 0:
        test_list.append(user_data.iloc[n_train+n_valid:])

df_train = pd.concat(train_list, ignore_index=True)
df_valid = pd.concat(valid_list, ignore_index=True)
df_test = pd.concat(test_list, ignore_index=True)

print(f"\n📊 Split Statistics:")
print(f"   Train: {len(df_train)} interactions, {df_train['user_id'].nunique()} users")
print(f"   Valid: {len(df_valid)} interactions, {df_valid['user_id'].nunique()} users")
print(f"   Test:  {len(df_test)} interactions, {df_test['user_id'].nunique()} users")

# Verify all users appear in all splits
train_users = set(df_train['user_id'])
valid_users = set(df_valid['user_id'])
test_users = set(df_test['user_id'])

print(f"\n✅ Verification:")
print(f"   Users in train only: {len(train_users - valid_users - test_users)}")
print(f"   Users in all three splits: {len(train_users & valid_users & test_users)}")

# -------------------------
# Save .train, .valid, .test
# -------------------------
inter_file_cols = ["user_id:token", "item_id:token", "label:float"]

df_train_renamed = df_train.copy()
df_train_renamed.columns = inter_file_cols
df_train_renamed.to_csv(TRAIN_FILE, sep='\t', index=False)
print(f"\n✅ Saved {TRAIN_FILE}")

df_valid_renamed = df_valid.copy()
df_valid_renamed.columns = inter_file_cols
df_valid_renamed.to_csv(VALID_FILE, sep='\t', index=False)
print(f"✅ Saved {VALID_FILE}")

df_test_renamed = df_test.copy()
df_test_renamed.columns = inter_file_cols
df_test_renamed.to_csv(TEST_FILE, sep='\t', index=False)
print(f"✅ Saved {TEST_FILE}")

print("\n" + "="*50)
print("✅ All files created successfully!")
print("="*50)
print(f"\nDataset location: {DATASET_DIR}")
print(f"Files created:")
print(f"  - {os.path.basename(USER_FILE)}")
print(f"  - {os.path.basename(ITEM_FILE)}")
print(f"  - {os.path.basename(TRAIN_FILE)}")
print(f"  - {os.path.basename(VALID_FILE)}")
print(f"  - {os.path.basename(TEST_FILE)}")
print("\n⚠️  NOTE:")
print("    - job_id renamed to item_id")
print("    - Only positive interactions (label=1) included")
print("    - Every user appears in train, validation, AND test sets")
print("    - Users with <3 interactions were filtered out")