"""Create an explicitly synthetic fixture for server smoke tests, never paper results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    directory = Path(args.output) / 'baby'
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError('Refusing to overwrite a nonempty data directory')
    rows = [(user, user * 5 + offset, 0 if offset < 3 else offset - 2)
            for user in range(12) for offset in range(5)]
    pd.DataFrame(rows, columns=['userID', 'itemID', 'x_label']).to_csv(directory / 'baby.inter', sep='\t', index=False)
    rng = np.random.default_rng(44)
    for name, dimension in [('image', 16), ('text', 12)]:
        np.save(directory / (name + '_feat.npy'), rng.normal(size=(60, dimension)).astype('float32'))
    (directory / 'SYNTHETIC_DATA.txt').write_text('Synthetic fixture. Do not report metrics as research results.\n')
    print(directory)
