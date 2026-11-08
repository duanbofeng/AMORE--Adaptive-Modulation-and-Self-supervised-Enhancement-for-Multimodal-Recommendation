"""Train-only interventions with explicit provenance for reviewer experiments."""
import hashlib
import numpy as np


def perturb_training(train, valid, test, config):
    rate = float(config['interaction_noise_rate'] or 0.)
    cold = float(config['cold_item_fraction'] or 0.)
    if not 0 <= rate <= 1 or not 0 <= cold < 1 or (rate and cold):
        raise ValueError('Use one intervention at a time; noise in [0,1], cold fraction in [0,1)')
    seed = int(config['intervention_seed'] if config['intervention_seed'] is not None else 314159)
    rng = np.random.default_rng(seed)
    frame = train.df.copy()
    uid, iid = train.uid_field, train.iid_field
    metadata = {'seed': seed, 'noise_added': 0, 'cold_items': []}
    if cold:
        items = np.sort(frame[iid].unique())
        selected = rng.choice(items, size=int(round(cold * len(items))), replace=False)
        frame = frame[~frame[iid].isin(selected)].copy()
        metadata['cold_items'] = sorted(map(int, selected))
    if rate:
        # Known validation/test positives are an exclusion list only; never train on them.
        forbidden = set()
        for split in (train, valid, test):
            forbidden.update(zip(split.df[uid].astype(int), split.df[iid].astype(int)))
        count = int(round(rate * len(frame)))
        observed_users = np.sort(frame[uid].unique())
        user_set = set(observed_users)
        n_items = train.get_item_num()
        capacity = len(observed_users) * n_items - sum(u in user_set for u, _ in forbidden)
        if count > capacity:
            raise ValueError('Not enough unobserved pairs to inject requested interaction noise')
        added = []
        attempts = 0
        while len(added) < count:
            u, i = int(rng.choice(observed_users)), int(rng.integers(n_items))
            attempts += 1
            if (u, i) not in forbidden:
                row = frame.iloc[0].copy()
                row[uid], row[iid] = u, i
                added.append(row)
                forbidden.add((u, i))
            if attempts > max(10000, count * 1000):
                raise RuntimeError('Noise sampling saturated; lower the requested ratio')
        if added:
            import pandas as pd
            frame = pd.concat([frame, pd.DataFrame(added)], ignore_index=True).astype(train.df.dtypes.to_dict())
        metadata['noise_added'] = len(added)
    pairs = frame[[uid, iid]].sort_values([uid, iid]).to_numpy(dtype=np.int64)
    metadata['train_pairs_sha256'] = hashlib.sha256(pairs.tobytes()).hexdigest()
    if len(frame) == 0:
        raise ValueError('Intervention removed every training interaction')
    return train.copy(frame), metadata
