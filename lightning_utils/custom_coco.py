import numpy as np
from pycocotools.cocoeval import COCOeval


class CustomCOCOeval(COCOeval):

    DEFAULT_DET_SUMMARY_SPECS = [
        dict(ap=1, iouThr=None,  areaRng='all',    maxDets=100, label='Average Precision  (AP) @[IoU=0.50:0.95, area=all]'),
        dict(ap=1, iouThr=None,   areaRng='small',  maxDets=100, label='Average Precision  (AP) @[IoU=0.50:0.95, area=small]'),
        dict(ap=1, iouThr=None,   areaRng='medium', maxDets=100, label='Average Precision  (AP) @[IoU=0.50:0.95,      area=medium]'),
        dict(ap=1, iouThr=None,   areaRng='large',  maxDets=100, label='Average Precision  (AP) @[IoU=0.50:0.95,      area=large]'),

        dict(ap=1, iouThr=0.5,   areaRng='all',    maxDets=100, label='Average Precision  (AP) @[IoU=0.50,      area=all]'),
        dict(ap=1, iouThr=0.75,  areaRng='all',    maxDets=100, label='Average Precision  (AP) @[IoU=0.75,      area=all]'),
        dict(ap=1, iouThr=0.25,  areaRng='small',  maxDets=100, label='Average Precision  (AP) @[IoU=0.25,      area=small]'),
        dict(ap=1, iouThr=0.5,   areaRng='small',  maxDets=100, label='Average Precision  (AP) @[IoU=0.50,      area=small]'),
        dict(ap=1, iouThr=0.5,   areaRng='medium', maxDets=100, label='Average Precision  (AP) @[IoU=0.50,      area=medium]'),
        dict(ap=1, iouThr=0.5,   areaRng='large',  maxDets=100, label='Average Precision  (AP) @[IoU=0.50,      area=large]'),
        
        dict(ap=0, iouThr=None,  areaRng='all',    maxDets=1,   label='Average Recall     (AR) @[IoU=0.50:0.95, maxDets=1]'),
        dict(ap=0, iouThr=None,  areaRng='all',    maxDets=10,  label='Average Recall     (AR) @[IoU=0.50:0.95, maxDets=10]'),
        dict(ap=0, iouThr=None,  areaRng='all',    maxDets=100, label='Average Recall     (AR) @[IoU=0.50:0.95, maxDets=100]'),
        dict(ap=0, iouThr=None,  areaRng='small',  maxDets=100, label='Average Recall     (AR) @[IoU=0.50:0.95, area=small]'),
        dict(ap=0, iouThr=None,   areaRng='medium', maxDets=100, label='Average Recall     (AR) @[IoU=0.50:0.95,area=medium]'),
        dict(ap=0, iouThr=None,   areaRng='large',  maxDets=100, label='Average Recall     (AR) @[IoU=0.50:0.95,area=large]'),
    ]

    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox', summary_specs=None):
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
        self.summary_specs = summary_specs or self.DEFAULT_DET_SUMMARY_SPECS

        # Record which thresholds are "standard" (>=0.5) for the iouThr=None mean.
        # Then permanently expand params.iouThrs to include any custom thresholds.
        self._standard_iou_thrs = self.params.iouThrs.copy()
        extra = {
            round(spec['iouThr'], 4)
            for spec in self.summary_specs
            if spec.get('iouThr') is not None
            and round(spec['iouThr'], 4) not in set(np.round(self.params.iouThrs, 4))
        }
        if extra:
            self.params.iouThrs = np.array(
                sorted(set(np.round(self.params.iouThrs, 4)) | extra)
            )

    def _summarize_one(self, ap=1, iouThr=None, areaRng='all', maxDets=100, label=None):
        p = self.params
        aind = [i for i, lbl in enumerate(p.areaRngLbl) if lbl == areaRng]
        mind = [i for i, d   in enumerate(p.maxDets)    if d   == maxDets]

        if not aind:
            raise ValueError(f"areaRng '{areaRng}' not found in params.areaRngLbl: {p.areaRngLbl}")
        if not mind:
            raise ValueError(f"maxDets {maxDets} not found in params.maxDets: {p.maxDets}")

        eval_iou_thrs = self.eval['params'].iouThrs

        if ap == 1:
            s = self.eval['precision']   # [T, R, K, A, M]
            if iouThr is not None:
                t = np.where(np.isclose(eval_iou_thrs, iouThr))[0]
                if len(t) == 0:
                    raise ValueError(f"iouThr {iouThr} not found in eval thresholds: {eval_iou_thrs}")
                s = s[t]
            else:
                # Average only over the original standard thresholds (>=0.5),
                # not any custom sub-0.5 thresholds we injected.
                t = np.where(np.isin(np.round(eval_iou_thrs, 4),
                                     np.round(self._standard_iou_thrs, 4)))[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            s = self.eval['recall']      # [T, K, A, M]
            if iouThr is not None:
                t = np.where(np.isclose(eval_iou_thrs, iouThr))[0]
                if len(t) == 0:
                    raise ValueError(f"iouThr {iouThr} not found in eval thresholds: {eval_iou_thrs}")
                s = s[t]
            else:
                t = np.where(np.isin(np.round(eval_iou_thrs, 4),
                                     np.round(self._standard_iou_thrs, 4)))[0]
                s = s[t]
            s = s[:, :, aind, mind]

        mean_s = np.mean(s[s > -1]) if len(s[s > -1]) else -1

        if label is None:
            iou_str = (
                f"{self._standard_iou_thrs[0]:.2f}:{self._standard_iou_thrs[-1]:.2f}"
                if iouThr is None else f"{iouThr:.2f}"
            )
            metric = "AP" if ap == 1 else "AR"
            label = f"{metric} @[IoU={iou_str}, area={areaRng}, maxDets={maxDets}]"

        print(f"{label:<55} = {mean_s:.3f}")
        return mean_s

    def summarize(self):
        if not self.eval:
            raise Exception("Please run accumulate() first.")

        self.stats = []
        self.stats_dict = {}

        for spec in self.summary_specs:
            val = self._summarize_one(
                ap=spec.get('ap', 1),
                iouThr=spec.get('iouThr'),
                areaRng=spec.get('areaRng', 'all'),
                maxDets=spec.get('maxDets', 100),
                label=spec.get('label'),
            )
            self.stats.append(val)
            if spec.get('label'):
                self.stats_dict[spec['label']] = val

        self.stats = np.array(self.stats)