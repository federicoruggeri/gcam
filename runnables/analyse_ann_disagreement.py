import pandas as pd
from pathlib import Path
from collections import defaultdict


def evaluate_annotations(df):
    annotations = [df[col].values for col in df.columns if col.startswith('Annotation')]
    assert len(annotations) == 2
    statistics = defaultdict(int)
    for a, b in zip(*annotations):
        a_set = set([label.strip() for label in a.split(',') if len(label)])
        b_set = set([label.strip() for label in b.split(',') if len(label)])
        intersection = a_set.intersection(b_set)
        if intersection == a_set or intersection == b_set:
            statistics['subsumes'] += 1
        elif not len(intersection):
            statistics['disjoint'] += 1
        else:
            statistics['intersection'] += 1

    counts = sum(statistics.values())
    assert counts == df.shape[0]

    return statistics


def evaluate_agreement(df):
    annotations = [df[col].values for col in df.columns if col.startswith('Annotation')]
    assert len(annotations) == 2

    agreement = df.Agreement.values
    statistics = defaultdict(int)
    for ann1, ann2, agr in zip(*annotations, agreement):
        ann1_set = set([label.strip() for label in ann1.split(',') if len(label)])
        ann2_set = set([label.strip() for label in ann2.split(',') if len(label)])
        agr_set = set([label.strip() for label in agr.split(',') if len(label)])

        union = ann1_set.union(ann2_set)

        if union == agr_set:
            statistics['union'] += 1
        elif agr_set == ann1_set or agr_set == ann2_set:
            statistics['one_option'] += 1
        elif agr_set.intersection(union) == union:
            statistics['subsumes'] += 1
        elif agr_set != ann1_set and agr_set != ann2_set:
            statistics['novel'] += 1
        else:
            statistics['unclassified'] += 1

    counts = sum(statistics.values())
    assert counts == df.shape[0]

    return statistics


if __name__ == '__main__':
    base_path = Path(__file__).parent.parent.resolve().joinpath('annotations')
    fb_df = pd.read_csv(base_path.joinpath('first_gcam_disagreement.csv'))
    sb_df = pd.read_csv(base_path.joinpath('second_gcam_disagreement.csv'))

    print(evaluate_annotations(fb_df))
    print(evaluate_annotations(sb_df))
    print(evaluate_agreement(fb_df))
    print(evaluate_agreement(sb_df))
