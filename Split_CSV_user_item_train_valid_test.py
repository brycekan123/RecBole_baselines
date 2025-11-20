import pandas as pd
import numpy as np
import random
import os

# -------------------------
# File paths
# -------------------------
CSV_FILE = "label_pref_no_text_onehotenc_k_or_more.csv"
DATASET_DIR = "dataset/mydata_presplit_user_ranking"

# npyfiles are in parent folder of RecBole_baselines
USER_EMB_FILE = "../npyfiles/user_embeddings.npy"
JOB_EMB_FILE  = "../npyfiles/job_embeddings.npy"
USER_MAP_FILE = "../npyfiles/user_id_to_index.npy"
JOB_MAP_FILE  = "../npyfiles/job_id_to_index.npy"

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
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
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
# Prepare .user dataframe
# -------------------------
print("\n" + "=" * 60)
print("PREPARING USER FILE")
print("=" * 60)
user_feature_cols = [col for col in user_cols if col != "user_id"]
df_users = df[["user_id"] + user_feature_cols].drop_duplicates("user_id")

user_file_cols = ["user_id:token"] + [f"{col}:float" for col in user_feature_cols]
df_users_renamed = df_users.copy()
df_users_renamed.columns = user_file_cols

print(f"✓ User dataframe prepared")
print(f"   Shape: {df_users_renamed.shape}")
print(f"   Unique users: {len(df_users)}")

# -------------------------
# Prepare .item dataframe (rename job_id to item_id)
# -------------------------
print("\n" + "=" * 60)
print("PREPARING ITEM FILE")
print("=" * 60)
item_feature_cols = [col for col in item_cols if col != "job_id"]
df_items = df[["job_id"] + item_feature_cols].drop_duplicates("job_id")

df_items = df_items.rename(columns={"job_id": "item_id"})

item_file_cols = ["item_id:token"] + [f"{col}:float" for col in item_feature_cols]
df_items_renamed = df_items.copy()
df_items_renamed.columns = item_file_cols

print(f"✓ Item dataframe prepared")
print(f"   Shape: {df_items_renamed.shape}")
print(f"   Unique items: {len(df_items)}")

# -------------------------
# Load embeddings
# -------------------------
print("\n" + "=" * 60)
print("LOADING EMBEDDINGS")
print("=" * 60)
user_emb = np.load(USER_EMB_FILE)  # shape: (5000, 6, 768)
job_emb = np.load(JOB_EMB_FILE)    # shape: (100, 768)

user_map = np.load(USER_MAP_FILE, allow_pickle=True).item()
job_map = np.load(JOB_MAP_FILE, allow_pickle=True).item()

# -------------------------
# COMPREHENSIVE SAFETY CHECKS
# -------------------------
print("\n" + "=" * 60)
print("SHAPE CHECKS")
print("=" * 60)
assert user_emb.ndim == 3 and user_emb.shape[2] == 768, "User embedding shape incorrect"
assert job_emb.ndim == 2 and job_emb.shape[1] == 768, "Job embedding shape incorrect"
print(f"✓ User embeddings: {user_emb.shape}")
print(f"✓ Job embeddings: {job_emb.shape}")
print(f"✓ User dataframe: {len(df_users_renamed)} rows")
print(f"✓ Item dataframe: {len(df_items_renamed)} rows")
print(f"✓ User mapping dict: {len(user_map)} entries")
print(f"✓ Job mapping dict: {len(job_map)} entries")

print(f"\nNote: User DF has {len(df_users_renamed)} rows, embeddings have {user_emb.shape[0]} rows")
print(f"      This is expected if you filtered users (DF <= embeddings)")
print(f"Note: Item DF has {len(df_items_renamed)} rows, embeddings have {job_emb.shape[0]} rows")
print(f"      This is expected if you filtered items (DF <= embeddings)")

# Check if dimensions match where they should
print("\n" + "=" * 60)
print("DIMENSION CONSISTENCY CHECKS")
print("=" * 60)
assert len(user_map) == user_emb.shape[0], f"User map size mismatch: {len(user_map)} vs {user_emb.shape[0]}"
assert len(job_map) == job_emb.shape[0], f"Job map size mismatch: {len(job_map)} vs {job_emb.shape[0]}"
print("✓ Mapping dicts cover all embeddings")

assert len(df_users_renamed) <= len(user_map), "User DF has more entries than mapping!"
assert len(df_items_renamed) <= len(job_map), "Item DF has more entries than mapping!"
print("✓ DataFrames are subsets of mappings (as expected after filtering)")

# Check for missing IDs
print("\n" + "=" * 60)
print("MISSING ID CHECKS")
print("=" * 60)
# Extract the actual user_id values (without :token suffix)
user_ids_in_df = [uid.replace(':token', '') if ':token' in str(uid) else str(uid) 
                  for uid in df_users_renamed.iloc[:, 0]]
item_ids_in_df = [iid.replace(':token', '') if ':token' in str(iid) else str(iid) 
                  for iid in df_items_renamed.iloc[:, 0]]

missing_users = [uid for uid in user_ids_in_df if uid not in user_map]
missing_jobs  = [jid for jid in item_ids_in_df if jid not in job_map]
if missing_users:
    raise ValueError(f"These user_ids are missing in mapping: {missing_users[:10]} ...")
if missing_jobs:
    raise ValueError(f"These item_ids are missing in mapping: {missing_jobs[:10]} ...")
print("✓ All user IDs in DF found in mapping")
print("✓ All item IDs in DF found in mapping")

# Check for index range validity
print("\n" + "=" * 60)
print("INDEX RANGE CHECKS")
print("=" * 60)
used_user_indices = [user_map[uid] for uid in user_ids_in_df]
used_job_indices = [job_map[jid] for jid in item_ids_in_df]

print(f"Used user indices range: {min(used_user_indices)} to {max(used_user_indices)}")
print(f"Used job indices range: {min(used_job_indices)} to {max(used_job_indices)}")
assert min(used_user_indices) >= 0 and max(used_user_indices) < user_emb.shape[0], "User indices out of bounds"
assert min(used_job_indices) >= 0 and max(used_job_indices) < job_emb.shape[0], "Job indices out of bounds"
print("✓ All used indices are within valid range")

# Check full mapping indices too
all_user_indices = list(user_map.values())
all_job_indices = list(job_map.values())
print(f"\nFull mapping user indices range: {min(all_user_indices)} to {max(all_user_indices)}")
print(f"Full mapping job indices range: {min(all_job_indices)} to {max(all_job_indices)}")

# Check for off-by-one: verify indices are 0-indexed and contiguous
print("\n" + "=" * 60)
print("OFF-BY-ONE CHECKS (for full mapping)")
print("=" * 60)
sorted_user_indices = sorted(all_user_indices)
sorted_job_indices = sorted(all_job_indices)
expected_user_range = list(range(len(all_user_indices)))
expected_job_range = list(range(len(all_job_indices)))

if sorted_user_indices == expected_user_range:
    print("✓ User indices are 0-indexed and contiguous (0, 1, 2, ..., N-1)")
else:
    print("⚠ WARNING: User indices are NOT contiguous!")
    print(f"  Expected: {expected_user_range[:10]} ...")
    print(f"  Got: {sorted_user_indices[:10]} ...")

if sorted_job_indices == expected_job_range:
    print("✓ Job indices are 0-indexed and contiguous (0, 1, 2, ..., N-1)")
else:
    print("⚠ WARNING: Job indices are NOT contiguous!")
    print(f"  Expected: {expected_job_range[:10]} ...")
    print(f"  Got: {sorted_job_indices[:10]} ...")

# Verify the mapping is bijective (one-to-one)
print("\n" + "=" * 60)
print("BIJECTION CHECKS")
print("=" * 60)
if len(set(all_user_indices)) == len(all_user_indices):
    print("✓ User mapping is one-to-one (no duplicate indices)")
else:
    print("⚠ WARNING: User mapping has duplicate indices!")
    
if len(set(all_job_indices)) == len(all_job_indices):
    print("✓ Job mapping is one-to-one (no duplicate indices)")
else:
    print("⚠ WARNING: Job mapping has duplicate indices!")

# Sample checks with actual IDs
print("\n" + "=" * 60)
print("SAMPLE MAPPING CHECKS (first 10 users and items in DF)")
print("=" * 60)
print("\nUser samples:")
for uid in user_ids_in_df[:10]:
    idx = user_map[uid]
    avg_emb = user_emb[idx].mean(axis=0)
    print(f"  {uid} -> index {idx}, emb[0:3] = {avg_emb[:3]}")

print("\nItem samples:")
for jid in item_ids_in_df[:10]:
    idx = job_map[jid]
    emb = job_emb[idx]
    print(f"  {jid} -> index {idx}, emb[0:3] = {emb[:3]}")

# Random samples
print("\n" + "=" * 60)
print("RANDOM SAMPLE CHECKS (5 random from each)")
print("=" * 60)
print("\nRandom user samples:")
for uid in random.sample(user_ids_in_df, min(5, len(user_ids_in_df))):
    idx = user_map[uid]
    avg_emb = user_emb[idx].mean(axis=0)
    print(f"  {uid} -> index {idx}, emb[0:3] = {avg_emb[:3]}")

print("\nRandom item samples:")
for jid in random.sample(item_ids_in_df, min(5, len(item_ids_in_df))):
    idx = job_map[jid]
    emb = job_emb[idx]
    print(f"  {jid} -> index {idx}, emb[0:3] = {emb[:3]}")

# Cross-check: verify same ID always maps to same index
print("\n" + "=" * 60)
print("CONSISTENCY CHECKS")
print("=" * 60)
test_users = random.sample(user_ids_in_df, min(3, len(user_ids_in_df)))
test_items = random.sample(item_ids_in_df, min(3, len(item_ids_in_df)))

print("Verifying mapping consistency (checking same ID multiple times):")
for uid in test_users:
    idx1 = user_map[uid]
    idx2 = user_map[uid]
    emb1 = user_emb[idx1].mean(axis=0)
    emb2 = user_emb[idx2].mean(axis=0)
    consistent = idx1 == idx2 and np.allclose(emb1, emb2)
    status = "✓" if consistent else "✗"
    print(f"  {status} User {uid}: index={idx1}, consistent={consistent}")

for jid in test_items:
    idx1 = job_map[jid]
    idx2 = job_map[jid]
    emb1 = job_emb[idx1]
    emb2 = job_emb[idx2]
    consistent = idx1 == idx2 and np.allclose(emb1, emb2)
    status = "✓" if consistent else "✗"
    print(f"  {status} Item {jid}: index={idx1}, consistent={consistent}")

# -------------------------
# MERGE EMBEDDINGS
# -------------------------
print("\n" + "=" * 60)
print("MERGING EMBEDDINGS")
print("=" * 60)
user_emb_avg = user_emb.mean(axis=1)
user_vectors = np.vstack([user_emb_avg[user_map[uid]] for uid in user_ids_in_df])
user_emb_cols = [f"user_embed_{i}:float" for i in range(user_vectors.shape[1])]
user_emb_df = pd.DataFrame(user_vectors, columns=user_emb_cols)
df_users_renamed = pd.concat([df_users_renamed.reset_index(drop=True), user_emb_df], axis=1)

item_vectors = np.vstack([job_emb[job_map[jid]] for jid in item_ids_in_df])
item_emb_cols = [f"item_embed_{i}:float" for i in range(item_vectors.shape[1])]
item_emb_df = pd.DataFrame(item_vectors, columns=item_emb_cols)
df_items_renamed = pd.concat([df_items_renamed.reset_index(drop=True), item_emb_df], axis=1)

print(f"✓ Added {len(user_emb_cols)} embedding columns to user dataframe (user_embed_0 to user_embed_767)")
print(f"✓ Added {len(item_emb_cols)} embedding columns to item dataframe (item_embed_0 to item_embed_767)")

# Final sanity check on merged data
print("\n" + "=" * 60)
print("FINAL MERGED DATA CHECKS")
print("=" * 60)
print(f"User DF shape: {df_users_renamed.shape}")
print(f"Item DF shape: {df_items_renamed.shape}")
print(f"User DF columns: {list(df_users_renamed.columns[:5])} ... {list(df_users_renamed.columns[-3:])}")
print(f"Item DF columns: {list(df_items_renamed.columns[:5])} ... {list(df_items_renamed.columns[-3:])}")
print("✓ Embeddings successfully added to dataframes")

# -------------------------
# Save .user and .item WITH EMBEDDINGS
# -------------------------
print("\n" + "=" * 60)
print("SAVING USER AND ITEM FILES")
print("=" * 60)
df_users_renamed.to_csv(USER_FILE, sep='\t', index=False)
print(f"✅ Saved {USER_FILE}")

df_items_renamed.to_csv(ITEM_FILE, sep='\t', index=False)
print(f"✅ Saved {ITEM_FILE}")

# -------------------------
# Split interactions per user into train/valid/test
# -------------------------
print("\n" + "=" * 60)
print("SPLITTING INTERACTIONS")
print("=" * 60)
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
print("\n" + "=" * 60)
print("SAVING INTERACTION FILES")
print("=" * 60)
inter_file_cols = ["user_id:token", "item_id:token", "label:float"]

df_train_renamed = df_train.copy()
df_train_renamed.columns = inter_file_cols
df_train_renamed.to_csv(TRAIN_FILE, sep='\t', index=False)
print(f"✅ Saved {TRAIN_FILE}")

df_valid_renamed = df_valid.copy()
df_valid_renamed.columns = inter_file_cols
df_valid_renamed.to_csv(VALID_FILE, sep='\t', index=False)
print(f"✅ Saved {VALID_FILE}")

df_test_renamed = df_test.copy()
df_test_renamed.columns = inter_file_cols
df_test_renamed.to_csv(TEST_FILE, sep='\t', index=False)
print(f"✅ Saved {TEST_FILE}")

print("\n" + "="*60)
print("✅ ALL FILES CREATED SUCCESSFULLY!")
print("="*60)
print(f"\nDataset location: {DATASET_DIR}")
print(f"Files created:")
print(f"  - {os.path.basename(USER_FILE)} (WITH EMBEDDINGS)")
print(f"  - {os.path.basename(ITEM_FILE)} (WITH EMBEDDINGS)")
print(f"  - {os.path.basename(TRAIN_FILE)}")
print(f"  - {os.path.basename(VALID_FILE)}")
print(f"  - {os.path.basename(TEST_FILE)}")
print("\n⚠️  NOTE:")
print("    - job_id renamed to item_id")
print("    - Only positive interactions (label=1) included")
print("    - Every user appears in train, validation, AND test sets")
print("    - Users with <3 interactions were filtered out")
print("    - User embeddings: user_embed_0 to user_embed_767")
print("    - Item embeddings: item_embed_0 to item_embed_767")