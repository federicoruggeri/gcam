from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_guideline_idx(item, guidelines):
    return guidelines[guidelines.Label == float(item)].index.values[0]


def get_true_labels(df, guidelines):
    y_true_classes = np.array([1 if label == 'sexist' else 0 for label in df.label_sexist.values])

    targets = [get_guideline_idx(item.split(' ')[0], guidelines) if item != 'none' else None for item in
               df.label_vector]
    y_true_guidelines = []
    for target_set in targets:
        target_mask = np.zeros((len(guidelines)))
        if target_set is not None:
            target_mask[target_set] = 1
        y_true_guidelines.append(target_mask.tolist())
    y_true_guidelines = np.array(y_true_guidelines)

    return y_true_classes, y_true_guidelines


def get_sap_info(sap_preds, y_true_classes, visualize=True):
    cm = confusion_matrix(y_pred=sap_preds['predictions'], y_true=y_true_classes)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm,
                                 display_labels=[0, 1])

    print(cm)
    if visualize:
        cmd.plot()
        plt.show()


def get_gcap_info(gcap_preds, y_true_classes, y_true_guidelines, visualize=True):
    # confusion matrices
    y_pred_classes, y_pred_guidelines = gcap_preds['predictions'], gcap_preds['guideline_predictions']
    y_true_guidelines = y_true_guidelines.argmax(axis=-1)
    y_pred_guidelines = y_pred_guidelines.argmax(axis=-1)

    cm_classes = confusion_matrix(y_pred=y_pred_classes, y_true=y_true_classes)
    cmd_classes = ConfusionMatrixDisplay(confusion_matrix=cm_classes,
                                         display_labels=[0, 1])
    cm_guidelines = confusion_matrix(y_pred=y_pred_guidelines, y_true=y_true_guidelines)
    cmd_guidelines = ConfusionMatrixDisplay(confusion_matrix=cm_guidelines,
                                            display_labels=np.arange(11))

    for row in cm_classes:
        print('& '.join([str(item) for item in row]))
    for row in cm_guidelines:
        print(' & '.join([str(item) for item in row]))

    if visualize:
        cmd_classes.plot()
        cmd_guidelines.plot()
        plt.show()

    # error types
    # A: C_x is wrong     --> edge cases?
    # B: C_x is correct and G_x is correct --> ideal case
    # C: C_x is correct and G_x is wrong --> semantic confounder?

    case_A = np.where((y_pred_classes != y_true_classes))[0]
    case_B = np.where((y_pred_classes == y_true_classes) & (y_pred_guidelines == y_true_guidelines))[0]
    case_C = np.where((y_pred_classes == y_true_classes) & (y_pred_guidelines != y_true_guidelines))[0]

    types_info = {
        'A (C_x wrong, G_x wrong)': case_A.shape[0],
        'B (C_x correct, G_x correct)': case_B.shape[0],
        'C (C_x correct, G_x wrong)': case_C.shape[0],
    }
    print(types_info)


if __name__ == '__main__':
    visualize = False

    base_path = Path(__file__).parent.parent.joinpath('results', 'edos')
    binary_path = base_path.joinpath('binary', 'test_predictions.npy')
    entail_path = base_path.joinpath('entail', 'test_predictions.npy')

    binary_info = np.load(binary_path, allow_pickle=True).item()
    entail_info = np.load(entail_path, allow_pickle=True).item()

    # top 5 binary seeds (in order): 33, 42, 15451, 1337
    # top 5 entail seeds (in order): 40000, 42, 2001, 33
    binary_seed = 33
    entail_seed = 40000

    edos_path = base_path.parent.with_name('data').joinpath('edos', 'test.csv')
    test_df = pd.read_csv(edos_path)
    guidelines = pd.read_csv(edos_path.with_name('guidelines.csv'))

    y_true_classes, y_true_guidelines = get_true_labels(df=test_df, guidelines=guidelines)

    get_sap_info(sap_preds=binary_info[binary_seed],
                 y_true_classes=y_true_classes,
                 visualize=visualize)
    get_gcap_info(gcap_preds=entail_info[entail_seed],
                  y_true_classes=y_true_classes,
                  y_true_guidelines=y_true_guidelines,
                  visualize=visualize)
