from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset',
                        '-d',
                        default='tos',
                        type=str,
                        nargs='?',
                        choices=['edos', 'tos'])
    args = parser.parse_args()
    dataset_name = args.dataset

    base_path = Path(__file__).parent.parent.resolve().joinpath('results', dataset_name)
    metrics_record = []
    for filename in base_path.rglob('*.npy'):
        if filename.name.startswith('metrics_'):
            folder_name = filename.parent.name
            metrics = np.load(filename.as_posix(), allow_pickle=True).item()
            if 'classes' in filename.name:
                metrics_record.append([folder_name, metrics, '-'])
            else:
                metrics_record.append([folder_name, metrics['classes'], metrics['guidelines']])

    metrics_df = pd.DataFrame.from_records(metrics_record, columns=['Name', 'C_x', 'G_x'])
    print(metrics_df.to_string())
