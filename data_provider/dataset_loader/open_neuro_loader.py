import random
import numpy as np

from data_provider.dataset_loader.base_loader import BaseLoader


class OpenNeuroLoader(BaseLoader):
    """Loader for a custom dataset in the common LEAD memmap/meta format.

    y.dat must contain exactly three columns:
        [disease_label, subject_id, sampling_rate]
    Labels should be consecutive integers starting at 0 (0, 1, 2, ...).
    """

    def _get_id_lists(self, args, data_list: np.ndarray, a: float, b: float):
        return get_id_list_openneuro(args, data_list, a, b)

    def _postprocess_labels(self, args, flag: str):
        # No label remapping: labels in y.dat are used as they are, which is
        # what --classify_choice multi_class expects.
        pass

def get_id_list_openneuro(args, data_list: np.ndarray, a=0.6, b=0.8
        ) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return subject-level TRAIN/VAL/TEST splits, stratified by label.

    data_list: shape (N_subject, 2) => [disease_label, subject_id]

    Subjects are split separately within each label (class) so that every
    class appears in train/val/test in roughly the same proportion. This works
    for any number of classes, so it does not need to be edited if your label
    set changes.

    Args:
        args : Command line arguments.
        data_list (np.ndarray): List with one row per subject, in the form [label, subject_id].
        a (float, optional): Train / Validation split point. Defaults to 0.6.
        b (float, optional): Validation / Testing split point. Defaults to 0.8.

    Raises:
        ValueError: cross_val must be "fixed" or "mccv".

    Returns:
        (list[int], list[int], list[int], list[int]): all subject ids, training ids, validation ids, testing ids
    """    
    subjects_ids = list(data_list[:, 1])

    if args.cross_val == "fixed": # to make tests reproducible
        random.seed(42)
    elif args.cross_val == "mccv": # Monte Carlo cross-validation
        random.seed(args.seed)
    else:
        raise ValueError(
            "Invalid cross_val for OpenNeuro. Please use fixed or mccv."
        )

    train_ids, val_ids, test_ids = [], [], []

    for label in np.unique(data_list[:, 0]):
        #grabbing subjecst with current disease label
        relevant_subjects = data_list[data_list[:, 0] == label]
        ids = list(relevant_subjects[:, 1])
        random.shuffle(ids)

        n = len(ids)

        # adding relevant split of current disease to each group
        train_ids += ids[: int(a * n)]
        val_ids += ids[int(a * n) : int(b * n)]
        test_ids += ids[int(b * n) :]

    return sorted(subjects_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)
