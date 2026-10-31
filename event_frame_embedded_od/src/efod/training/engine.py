from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader


def _move(images,targets,device):
    images=[x.to(device,non_blocking=True) for x in images]
    targets=[{k:(v.to(device) if torch.is_tensor(v) else v) for k,v in t.items()} for t in targets]
    return images,targets


def make_optimizer_scheduler(model,epochs:int,lr=5e-3,weight_decay=1e-2):
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs))
    return opt,sch


def _step(model,images,targets,optimizer,grad_clip=None):
    optimizer.zero_grad(set_to_none=True); losses=model(images,targets); loss=sum(losses.values()); loss.backward()
    if grad_clip is not None: torch.nn.utils.clip_grad_norm_(model.parameters(),float(grad_clip))
    optimizer.step()
    return float(loss.detach()),{k:float(v.detach()) for k,v in losses.items()}


def train_ann_epoch(model,loader:DataLoader,optimizer,device='cuda',grad_clip=None):
    model.train(); total=0.; updates=0; sums={'bbox_regression':0.,'classification':0.}
    for images,targets,_ in loader:
        images,targets=_move(images,targets,device)
        if all(t['boxes'].numel()==0 for t in targets): continue
        val,parts=_step(model,images,targets,optimizer,grad_clip); total+=val; updates+=1
        for k in sums: sums[k]+=parts.get(k,0.)
    return {'loss':total/max(1,updates),'updates':updates,**{f'{k}_loss':v/max(1,updates) for k,v in sums.items()}}


def train_snn_reset_epoch(model,loader,optimizer,device='cuda',grad_clip=None):
    """Independent-window SNN training baseline: reset membrane state before every batch."""
    model.train(); total=0.; updates=0; sums={'bbox_regression':0.,'classification':0.}
    for images,targets,_ in loader:
        model.reset_state(); images,targets=_move(images,targets,device)
        if all(t['boxes'].numel()==0 for t in targets): continue
        val,parts=_step(model,images,targets,optimizer,grad_clip); model.detach_state(); total+=val; updates+=1
        for k in sums: sums[k]+=parts.get(k,0.)
    return {'loss':total/max(1,updates),'updates':updates,**{f'{k}_loss':v/max(1,updates) for k,v in sums.items()}}


def train_snn_no_reset_epoch(model,loader:DataLoader,optimizer,device='cuda',max_unlabeled_steps:int=5,grad_clip=None):
    """Cordone-style continuous 50-ms training with state continuity + truncated BPTT.

    batch_size=1 and shuffle=False are required. State is reset only at sequence boundaries.
    Windows without GT still update membrane potentials but do not create a background loss.
    A target window triggers loss/backward/update and state detach. A long unlabeled run is
    detached after ``max_unlabeled_steps`` to cap graph memory, matching the thesis principle.
    """
    model.train(); model.reset_state(); current=None; since_detach=0; total=0.; updates=0
    sums={'bbox_regression':0.,'classification':0.}
    for images,targets,meta in loader:
        if len(images)!=1: raise ValueError('No-reset SNN training requires batch_size=1')
        seq=meta[0]['sequence']
        if seq!=current: model.reset_state(); current=seq; since_detach=0
        images,targets=_move(images,targets,device); has_target=targets[0]['boxes'].numel()>0
        if has_target:
            val,parts=_step(model,images,targets,optimizer,grad_clip); model.detach_state(); since_detach=0
            total+=val; updates+=1
            for k in sums: sums[k]+=parts.get(k,0.)
        else:
            model.update_state_only(images[0]); since_detach+=1
            if since_detach>=max_unlabeled_steps: model.detach_state(); since_detach=0
    return {'loss':total/max(1,updates),'updates':updates,**{f'{k}_loss':v/max(1,updates) for k,v in sums.items()}}


def save_checkpoint(path,model,optimizer=None,scheduler=None,epoch=None,extra=None):
    payload={'model':model.state_dict(),'config':vars(model.cfg),'epoch':epoch,'extra':extra or {}}
    if optimizer is not None: payload['optimizer']=optimizer.state_dict()
    if scheduler is not None: payload['scheduler']=scheduler.state_dict()
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)


def load_checkpoint(path,model,optimizer=None,scheduler=None,map_location='cpu'):
    ckpt=torch.load(path,map_location=map_location); model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt

def load_model_from_checkpoint(path,device='cpu'):
    """Rebuild a detector from the serialized ModelConfig and load its weights."""
    from ..config import ModelConfig
    from ..models.detector import DetectionSystem
    ckpt=torch.load(path,map_location=device); cfg=ModelConfig(**ckpt['config']); model=DetectionSystem(cfg).to(device); model.load_state_dict(ckpt['model']); return model,ckpt

@torch.no_grad()
def probe_snn_activity(model,dataset,device='cuda',max_windows:int=32,start_fraction:float=.25):
    """Measure chronological Event-SNN activity on a short contiguous dataset segment.

    This is a training diagnostic only.  It deliberately runs ``model.eval()`` so the
    reported rate uses the BN running statistics that would be used for inference.
    Membrane state is reset only at sequence boundaries, matching no-reset inference.
    """
    from ..models.neurons import ParametricLIF
    if max_windows <= 0 or len(dataset) == 0:
        return {'probe_spike_rate':float('nan'),'probe_spike_sparsity':float('nan'),'probe_windows':0}
    was_training=model.training; model.eval(); model.reset_state()
    start=min(max(0,int(len(dataset)*float(start_fraction))),max(0,len(dataset)-1))
    end=min(len(dataset),start+int(max_windows)); total_spikes=0; total_neurons=0; current=None
    layer_sums={}
    for idx in range(start,end):
        image,_,meta=dataset[idx]
        seq=meta['sequence']
        if seq!=current:
            model.reset_state(); current=seq
        x=image.unsqueeze(0).to(device,non_blocking=True)
        model.forward_raw(x)
        for name,m in model.named_modules():
            if isinstance(m,ParametricLIF):
                total_spikes += int(m.last_spike_count); total_neurons += int(m.last_neuron_count)
                rec=layer_sums.setdefault(name,[0,0]); rec[0]+=int(m.last_spike_count); rec[1]+=int(m.last_neuron_count)
    model.reset_state()
    if was_training: model.train()
    rate=total_spikes/max(1,total_neurons)
    return {
        'probe_spike_rate':rate,
        'probe_spike_sparsity':1-rate,
        'probe_windows':max(0,end-start),
        'probe_layer_rates':{k:(v[0]/max(1,v[1])) for k,v in layer_sums.items()},
    }
