from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn
from ..models.blocks import PreActSpikingBlock

@dataclass
class Thesis45nmConfig:
    add_pj:float=.1; mul_pj:float=3.1; word_bits:int=32; fifo_elements:int=1000
    sram_points:tuple[tuple[int,float],...]=((8*1024,10.),(32*1024,20.),(1024*1024,100.))

def sram_access_pj(size_bytes,cfg):
    pts=cfg.sram_points; x=max(1,int(size_bytes))
    if x<=pts[0][0]: return pts[0][1]
    for (x0,y0),(x1,y1) in zip(pts[:-1],pts[1:]):
        if x<=x1: return y0+(y1-y0)*(x-x0)/(x1-x0)
    return pts[-1][1]
def _bytes(n,b): return int(math.ceil(n*b/8))
def _record(name,conv,inp,out):
    x=inp[0]; b,cin,hin,win=x.shape; _,cout,hout,wout=out.shape; kh,kw=conv.kernel_size; sh,sw=conv.stride
    return dict(name=name,conv=conv,batch=b,cin=cin,hin=hin,win=win,cout=cout,hout=hout,wout=wout,kh=kh,kw=kw,sh=sh,sw=sw)

def estimate_thesis_energy(model,input_tensor,cfg=None,reset_state=True):
    """Cordone/Lemaire Ch.6-style E=E_mem+E_ops+E_addr for one forward window.

    Assumptions are intentionally explicit: 32-bit data, local SRAM, 1000-word SNN FIFO,
    BN fused into adjacent conv at inference, hard reset, one SNN timestep/window. Spiking
    backbone/extras use measured theta_in/theta_out; analog SSD prediction heads are dense.
    """
    cfg=cfg or Thesis45nmConfig(); conv2block={id(b.conv):b for b in model.modules() if isinstance(b,PreActSpikingBlock)}
    rec=[]; hooks=[]
    for name,m in model.named_modules():
        if isinstance(m,nn.Conv2d): hooks.append(m.register_forward_hook(lambda mod,inp,out,n=name:rec.append(_record(n,mod,inp,out))))
    if reset_state and hasattr(model,'reset_state'): model.reset_state()
    with torch.no_grad(): model.forward_raw(input_tensor)
    for h in hooks: h.remove()
    keys=['ann_synaptic_macs','ann_synaptic_accs','snn_synaptic_macs','snn_synaptic_accs','ann_address_macs','ann_address_accs','snn_address_macs','snn_address_accs',
          'in_reads','parameter_reads','potential_reads','output_writes','potential_writes']
    totals={k:0 for k in keys}; totals.update(memory_energy_pj=0.,synaptic_energy_pj=0.,address_energy_pj=0.)
    rows=[]; bits=cfg.word_bits; fifo_bytes=_bytes(cfg.fifo_elements,bits)
    for r in rec:
        conv=r['conv']; cin,hin,win=r['cin'],r['hin'],r['win']; cout,hout,wout=r['cout'],r['hout'],r['wout']; kh,kw,sh,sw=r['kh'],r['kw'],r['sh'],r['sw']
        outn=cout*hout*wout; dense=cin*kh*kw*cout*hout*wout; sp=conv2block.get(id(conv)); pot_reads=pot_writes=0
        if sp is None:
            mac=dense; acc=outn; amac=0; aacc=cin*hin*win+outn+cout*kh*kw
            in_reads=dense; param_reads=(cin*kh*kw+1)*outn; out_writes=outn
            mem=in_reads*sram_access_pj(_bytes(cin*hin*win,bits),cfg)+param_reads*sram_access_pj(_bytes(cin*kh*kw*cout+cout,bits),cfg)+out_writes*sram_access_pj(_bytes(outn,bits),cfg)
            totals['ann_synaptic_macs']+=mac; totals['ann_synaptic_accs']+=acc; totals['ann_address_macs']+=amac; totals['ann_address_accs']+=aacc
            kind='ANN/dense'; ti=to=None
        else:
            ti=int(sp.last_theta_in); to=int(sp.last_theta_out)
            # Eq. 6.2: leakage MAC per output neuron for LIF/PLIF, spike-triggered ACCs,
            # plus per-output timestep update. Hard reset => no theta_out reset ACC term.
            mac=outn; acc=ti*math.ceil(kh/sh)*math.ceil(kw/sw)*cout+outn
            amac=ti*2; aacc=ti*cout*kh*kw
            in_reads=ti; param_reads=ti*cout*kh*kw+outn; pot_reads=ti*cout*kh*kw+outn; out_writes=to; pot_writes=ti*cout*kh*kw+outn
            mem=(in_reads*sram_access_pj(fifo_bytes,cfg)+param_reads*sram_access_pj(_bytes(cin*kh*kw*cout+cout,bits),cfg)+
                 pot_reads*sram_access_pj(_bytes(outn,bits),cfg)+out_writes*sram_access_pj(fifo_bytes,cfg)+pot_writes*sram_access_pj(_bytes(outn,bits),cfg))
            totals['snn_synaptic_macs']+=mac; totals['snn_synaptic_accs']+=acc; totals['snn_address_macs']+=amac; totals['snn_address_accs']+=aacc; kind='SNN/spiking'
        syn=(cfg.add_pj+cfg.mul_pj)*mac+cfg.add_pj*acc; addr=(cfg.add_pj+cfg.mul_pj)*amac+cfg.add_pj*aacc
        for k,v in [('in_reads',in_reads),('parameter_reads',param_reads),('potential_reads',pot_reads),('output_writes',out_writes),('potential_writes',pot_writes)]: totals[k]+=v
        totals['memory_energy_pj']+=mem; totals['synaptic_energy_pj']+=syn; totals['address_energy_pj']+=addr
        rows.append({'name':r['name'],'kind':kind,'dense_macs':dense,'synaptic_macs':mac,'synaptic_accs':acc,'address_macs':amac,'address_accs':aacc,'theta_in':ti,'theta_out':to,
                     'in_reads':in_reads,'parameter_reads':param_reads,'potential_reads':pot_reads,'output_writes':out_writes,'potential_writes':pot_writes,
                     'memory_energy_pj':mem,'synaptic_energy_pj':syn,'address_energy_pj':addr,'total_energy_pj':mem+syn+addr})
    total=totals['memory_energy_pj']+totals['synaptic_energy_pj']+totals['address_energy_pj']; out=dict(totals)
    out.update(thesis_memory_uJ=totals['memory_energy_pj']/1e6,thesis_synaptic_uJ=totals['synaptic_energy_pj']/1e6,thesis_address_uJ=totals['address_energy_pj']/1e6,thesis_total_uJ=total/1e6,
               thesis_energy_model='Cordone/Lemaire Ch.6 45nm reference; BN-fused; local SRAM; SNN FIFO=1000; 32-bit; hard reset; T=1',layer_rows=rows)
    return out

def estimate_thesis_energy_sequence(model,inputs,cfg=None,state_mode='stream'):
    """Average Chapter-6 estimate over a list of windows/frames, preserving SNN state in stream mode."""
    if not inputs: raise ValueError('inputs is empty')
    if hasattr(model,'reset_state'): model.reset_state()
    vals=[]; layers=[]
    for i,x in enumerate(inputs):
        if state_mode=='reset' and hasattr(model,'reset_state'): model.reset_state()
        r=estimate_thesis_energy(model,x,cfg,reset_state=False); layers.append(r.pop('layer_rows')); vals.append(r)
    numeric={}
    for k in vals[0]:
        if isinstance(vals[0][k],(int,float)): numeric[k]=sum(float(v[k]) for v in vals)/len(vals)
    numeric['thesis_sequence_windows']=len(vals); numeric['thesis_state_mode']=state_mode; numeric['layer_rows_sequence']=layers
    return numeric
