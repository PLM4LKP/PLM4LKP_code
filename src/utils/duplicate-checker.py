import pandas as pd

# use set to remove duplicate data
# so the size of set can show whether there is duplicate data
def get_pairs_set(data_path):
    pairs_set = set()

    dataset = pd.read_csv(data_path, lineterminator="\n")
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        q1id = row['q1_Id']
        q2id = row['q2_Id']

        # remove order
        # 1-2 and 2-1 are duplicate data
        p = min(q1id, q2id)
        n = max(q1id, q2id)

        pairs_set.add(f'{p}-{n}')

    return pairs_set

train_pairs_set = get_pairs_set('data/raw/medium_link_prediction_noClue_shuffled_train.csv')
test_pairs_set = get_pairs_set('data/raw/medium_link_prediction_noClue_shuffled_test.csv')

# check size
assert len(train_pairs_set) == 32000
assert len(test_pairs_set) == 8000
assert len(train_pairs_set | test_pairs_set) == 40000

print('ok, no duplicate data')