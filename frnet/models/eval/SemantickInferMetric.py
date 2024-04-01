# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence
import yaml
import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
import mmengine
from mmdet3d.evaluation import seg_eval
from mmdet3d.registry import METRICS
import pdb
@METRICS.register_module()
class SemantickInferMertric(BaseMetric):
    
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 result_path:str = None,
                 result_start_index:int = 0,
                 conf:str = None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        self.result_path = result_path
        self.result_start_index = result_start_index
        self.current_start_index = self.result_start_index
        self.limit=[921,1061,3281,631,1901,1731,491,1801,4981,831,2721]#the length of every test set seq
        self.limit_id = 0 #count 
        self.scene_id = 0 #seq id
        super(SemantickInferMertric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.conf = conf
        if self.conf :
            with open(self.conf) as f:
                self.conf = yaml.safe_load(f)
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        self.results.append((0, 0))

        # label inv
        # pdb.set_trace()
        # pdb.set_trace()
        pred = data_samples[0]['pred_pts_seg']['pts_semantic_mask'] #labels
        map_inv = self.dataset_meta['learning_map_inv'] #inv mapping
        pred[pred == 19] += 99 #unlabeled or ignore
        pred += 1 #[0,18] -> [1,19]
        for i in map_inv: 
            pred[pred==i] = map_inv[i]+1000  #avoid the pred label in [1-19] be mapped twice 
        pred[pred!=119]-=1000
        pred[pred==119] = 0 #unlabel 
        pred.cpu().numpy().astype(np.int32).tofile(f'/mnt/storage/dataset/semanticKitti/dataset/FRNet/sequences/{self.scene_id+11}/predictions/{self.limit_id:06}.label')
        print(f'finsh the {self.scene_id+11}/predictions/{self.limit_id:06}.label')
        self.limit_id+=1
        if self.limit_id == self.limit[self.scene_id]: #next sequence
            self.scene_id += 1
            self.limit_id = 0
        print(111)
        # pdb.set_trace()

        # output to label file
    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

       

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        ret_dict = dict()

        return ret_dict
