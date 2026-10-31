from __future__ import annotations

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection.ssd import DefaultBoxGenerator, SSDHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import box_iou, batched_nms, clip_boxes_to_image, sigmoid_focal_loss

from ..config import ModelConfig
from .backbones import STVGGBackbone, ResCatBackbone, SSDFeatureBackbone
from .neurons import reset_snn_state, detach_snn_state


class DetectionSystem(nn.Module):
    """Standalone six-scale SSD detector around the thesis-validated backbones.

    The ST-VGG/ResCat core and reduced SSD feature topology follow Cordone Ch. 5.
    The loss/anchor/postprocess implementation is an independent torchvision-based
    implementation for DSEC's eight foreground classes. It is not claimed to be a
    byte-identical copy of the original repository's detector-head utility code.
    """
    def __init__(self,cfg:ModelConfig):
        super().__init__(); self.cfg=cfg; spec=cfg.spec
        common=dict(C=spec['C'],pipeline=cfg.pipeline,input_channels=cfg.input_channels,
                    init_tau=cfg.plif_init_tau,threshold=cfg.plif_threshold)
        core=STVGGBackbone(**common) if spec['family']=='stvgg' else ResCatBackbone(**common)
        self.backbone=SSDFeatureBackbone(core,cfg.pipeline,cfg.extra_channels,cfg.plif_init_tau,cfg.plif_threshold)
        self.anchor_generator=DefaultBoxGenerator([list(x) for x in cfg.aspect_ratios],min_ratio=cfg.min_ratio,max_ratio=cfg.max_ratio,clip=True)
        num_anchors=self.anchor_generator.num_anchors_per_location()
        # Foreground-only sigmoid focal head: no explicit background channel.
        self.head=SSDHead(self.backbone.out_channels,num_anchors,cfg.foreground_classes)
        self.box_coder=det_utils.BoxCoder(weights=(10.0,10.0,5.0,5.0))
        self.proposal_matcher=det_utils.SSDMatcher(0.5)
        self.score_thresh=cfg.score_thresh; self.nms_thresh=cfg.nms_thresh
        self.detections_per_img=cfg.detections_per_img; self.topk_candidates=200
        self.focal_alpha=0.25; self.focal_gamma=2.0

    def _stack(self,images):
        if torch.is_tensor(images): return images.unsqueeze(0) if images.ndim==3 else images
        if not images: raise ValueError('images list is empty')
        shapes=[tuple(x.shape) for x in images]
        if len(set(shapes))!=1: raise ValueError(f'Fixed-size batch required; got {shapes}')
        return torch.stack(images,0)

    def forward_raw(self,batch):
        feats=self.backbone(batch); fl=list(feats.values()) if isinstance(feats,(dict,OrderedDict)) else list(feats)
        return feats,self.head(fl)

    def _anchors(self,batch,features):
        shapes=[(batch.shape[-2],batch.shape[-1])]*batch.shape[0]
        return self.anchor_generator(ImageList(batch,shapes),list(features.values()))

    def compute_loss(self,targets,head_outputs,anchors):
        bbox_reg=head_outputs['bbox_regression']; cls_logits=head_outputs['cls_logits']
        bl=[]; cl=[]; total_fg=0
        for target,reg_i,cls_i,anchors_i in zip(targets,bbox_reg,cls_logits,anchors):
            gt_boxes=target['boxes']; gt_labels=target['labels']
            matched=(torch.full((anchors_i.shape[0],),-1,device=anchors_i.device,dtype=torch.long)
                     if gt_boxes.numel()==0 else self.proposal_matcher(box_iou(gt_boxes,anchors_i)))
            fg=torch.where(matched>=0)[0]; total_fg+=int(fg.numel())
            if fg.numel():
                reg_target=self.box_coder.encode_single(gt_boxes[matched[fg]],anchors_i[fg])
                bl.append(F.smooth_l1_loss(reg_i[fg],reg_target,reduction='sum'))
            else: bl.append(reg_i.sum()*0.0)
            one_hot=torch.zeros_like(cls_i)
            if fg.numel():
                labels=gt_labels[matched[fg]].long()
                if labels.min()<0 or labels.max()>=self.cfg.foreground_classes:
                    raise ValueError(f'Labels must be 0..{self.cfg.foreground_classes-1}')
                one_hot[fg,labels]=1.0
            cl.append(sigmoid_focal_loss(cls_i,one_hot,alpha=self.focal_alpha,gamma=self.focal_gamma,reduction='sum'))
        norm=max(1,total_fg)
        return {'bbox_regression':torch.stack(bl).sum()/norm,'classification':torch.stack(cl).sum()/norm}

    def postprocess(self,head_outputs,anchors,image_shapes):
        results=[]
        for logits,reg,anc,shape in zip(head_outputs['cls_logits'],head_outputs['bbox_regression'],anchors,image_shapes):
            boxes=clip_boxes_to_image(self.box_coder.decode_single(reg,anc),shape); scores_all=torch.sigmoid(logits)
            bb=[]; ss=[]; ll=[]
            for c in range(self.cfg.foreground_classes):
                s=scores_all[:,c]; keep=torch.where(s>self.score_thresh)[0]
                if not keep.numel(): continue
                s=s[keep]; k=min(self.topk_candidates,s.numel()); s,idx=s.topk(k); keep=keep[idx]
                bb.append(boxes[keep]); ss.append(s); ll.append(torch.full((k,),c,dtype=torch.long,device=s.device))
            if not ss:
                results.append({'boxes':boxes.new_zeros((0,4)),'scores':logits.new_zeros((0,)),
                                'labels':torch.zeros((0,),dtype=torch.long,device=logits.device)}); continue
            b=torch.cat(bb); s=torch.cat(ss); l=torch.cat(ll); keep=batched_nms(b,s,l,self.nms_thresh)[:self.detections_per_img]
            results.append({'boxes':b[keep],'scores':s[keep],'labels':l[keep]})
        return results

    def forward(self,images,targets=None):
        batch=self._stack(images)
        if batch.shape[-2:]!=(self.cfg.image_height,self.cfg.image_width):
            raise ValueError(f'Expected {(self.cfg.image_height,self.cfg.image_width)}, got {tuple(batch.shape[-2:])}')
        features,head=self.forward_raw(batch); anchors=self._anchors(batch,features)
        if self.training:
            if targets is None: raise ValueError('targets required in training mode')
            return self.compute_loss(targets,head,anchors)
        return self.postprocess(head,anchors,[(self.cfg.image_height,self.cfg.image_width)]*batch.shape[0])

    def update_state_only(self,image):
        if self.cfg.pipeline!='event_snn': return
        if image.ndim==3: image=image.unsqueeze(0)
        _=self.backbone(image)

    def reset_state(self): reset_snn_state(self)
    def detach_state(self): detach_snn_state(self)
    def count_parameters(self): return sum(p.numel() for p in self.parameters())


def build_model(variant='vgg32',pipeline='event_snn',**kwargs):
    return DetectionSystem(ModelConfig(variant=variant,pipeline=pipeline,**kwargs))
