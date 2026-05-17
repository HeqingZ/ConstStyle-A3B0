import os
import os.path as osp
import numpy as np
from sklearn.model_selection import train_test_split

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class RadarFrequency(DatasetBase):
    """6-class FMCW radar frequency domain generalization dataset.

    Expected directory structure:

    domainGenDataset_core_movements/
        10GHz/
            Away/
            Bend/
            Kneel/
            Pick/
            Sit/
            Towards/
        24GHz/
            ...
        77GHz/
            ...
    """

    dataset_dir = "domainGenDataset_core_movements"
    domains = ["10GHz", "24GHz", "77GHz"]
    class_names = ["Away", "Bend", "Kneel", "Pick", "Sit", "Towards"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS,
            cfg.DATASET.TARGET_DOMAINS,
        )

        train, val = self._read_train_val(cfg.DATASET.SOURCE_DOMAINS, cfg.SEED)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="test")

        self._print_stats("Train", train)
        self._print_stats("Val", val)
        self._print_stats("Test", test)

        super().__init__(train_x=train, val=val, test=test)

    def _read_train_val(self, input_domains, seed):
        items = self._read_data(input_domains, split="train")
        labels = [item.label for item in items]

        train, val = train_test_split(
            items,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )

        return train, val

    def _read_data(self, input_domains, split):
        items = []

        class_to_label = {c: i for i, c in enumerate(self.class_names)}

        for domain_idx, domain_name in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, domain_name)

            if not osp.isdir(domain_dir):
                raise FileNotFoundError(f"Domain directory not found: {domain_dir}")

            # Keep test domain ids separated from source domain ids,
            # following the PACS convention in this repo.
            if split == "test":
                mapped_domain = domain_idx + 10
            else:
                mapped_domain = domain_idx

            for class_name in self.class_names:
                class_dir = osp.join(domain_dir, class_name)

                if not osp.isdir(class_dir):
                    raise FileNotFoundError(f"Class directory not found: {class_dir}")

                label = class_to_label[class_name]

                for fname in sorted(os.listdir(class_dir)):
                    if fname.startswith("."):
                        continue

                    impath = osp.join(class_dir, fname)

                    if not osp.isfile(impath):
                        continue

                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=mapped_domain,
                        classname=class_name,
                    )
                    items.append(item)

        return items

    def _print_stats(self, name, data):
        domains = [item.domain for item in data]
        labels = [item.label for item in data]

        domain_values, domain_counts = np.unique(domains, return_counts=True)
        label_values, label_counts = np.unique(labels, return_counts=True)

        print(
            f"{name} dataset statistics | "
            f"Domain {domain_values} - count {domain_counts} | "
            f"Class {label_values} - count {label_counts}"
        )
