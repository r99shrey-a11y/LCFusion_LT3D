"""ExcludingClassBalancedDataset — a thin subclass of mmengine's
ClassBalancedDataset that forces the repeat factor of specified category
NAMES to 1.0 (i.e. excludes them from oversampling boost), while every other
category is boosted exactly as in the stock implementation.

Purpose: isolate whether OVERSAMPLING ITSELF (repeating barrier images) is
the cause of barrier's fused-AP collapse under the curated multi-part dataset
(see thesis_context.md, Aug 10 diagnostic session), independent of the
"more/different data" variable already tested in dino_nuscenes_curated_cbd030.py.

All logic is inherited from mmengine.dataset.dataset_wrapper.ClassBalancedDataset
except _get_repeat_factors, which is overridden to zero out excluded classes'
contribution to r(c) before the per-image max-over-classes step (still uses
math.ceil() exactly as the base class, so results are directly comparable).

Registered as 'ExcludingClassBalancedDataset' in mmengine's DATASETS registry
(same registry the base class inherits its @DATASETS.register_module() from).
Import via custom_imports in the config; train_dino.py adds ~/LCFusion_LT3D to
sys.path so this module is importable.
"""
import math
from typing import List, Union

from mmengine.dataset import BaseDataset
from mmengine.dataset.dataset_wrapper import ClassBalancedDataset
from mmengine.registry import DATASETS


@DATASETS.register_module()
class ExcludingClassBalancedDataset(ClassBalancedDataset):
    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 oversample_thr: float,
                 exclude_categories: List[str] = (),
                 lazy_init: bool = False):
        self.exclude_categories = set(exclude_categories)
        self._exclude_cat_ids = None  # resolved lazily once metainfo/classes exist
        super().__init__(dataset, oversample_thr, lazy_init)

    def _resolve_exclude_ids(self):
        if self._exclude_cat_ids is not None:
            return self._exclude_cat_ids
        classes = self.dataset.metainfo.get('classes', [])
        self._exclude_cat_ids = {
            i for i, name in enumerate(classes) if name in self.exclude_categories
        }
        return self._exclude_cat_ids

    def _get_repeat_factors(self, dataset: BaseDataset,
                            repeat_thr: float) -> List[float]:
        """Identical to the base class, except categories in
        `self.exclude_categories` are removed from category_repeat before
        the per-image max step — so images are never duplicated on account
        of an excluded category, even if they also contain a NON-excluded
        boosted category (in which case the non-excluded category's repeat
        factor still applies normally)."""
        exclude_ids = self._resolve_exclude_ids()

        category_freq: dict = {}
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                category_freq[cat_id] = category_freq.get(cat_id, 0.0) + 1
        for k, v in category_freq.items():
            assert v > 0, f'category {k} does not contain any images'
            category_freq[k] = v / num_images

        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
            if cat_id not in exclude_ids          # <-- the only change vs base class
        }

        repeat_factors = []
        for idx in range(num_images):
            repeat_factor: float = 1.0
            cat_ids = set(self.dataset.get_cat_ids(idx)) - exclude_ids
            if len(cat_ids) != 0:
                repeat_factor = max(
                    {category_repeat[cat_id] for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors
