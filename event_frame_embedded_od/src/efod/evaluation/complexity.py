from __future__ import annotations

import io,time,statistics
import torch
from torch import nn
from ..models.neurons import ParametricLIF


def serialized_state_bytes(model):
    b=io.BytesIO(); torch.save(model.state_dict(),b); return len(b.getvalue())

def parameter_summary(model):
    p=sum(x.numel() for x in model.parameters()); t=sum(x.numel() for x in model.parameters() if x.requires_grad)
    return {'params':p,'trainable_params':t,'fp32_param_mib':p*4/2**20,'fp16_param_mib':p*2/2**20,'state_dict_mib':serialized_state_bytes(model)/2**20}

def membrane_state_mib(model):
    n=0; bytes_=0
    for m in model.modules():
        if isinstance(m,ParametricLIF) and m.v is not None:
            n+=m.v.numel(); bytes_+=m.v.numel()*m.v.element_size()
    return {'snn_state_elements':n,'snn_state_mib':bytes_/2**20}

def dense_conv_macs(model,input_tensor):
    rec=[]; hooks=[]
    def hook(name,m,inp,out):
        kh,kw=m.kernel_size; cin=m.in_channels//m.groups; b,cout,h,w=out.shape
        rec.append({'name':name,'macs':int(b*cout*h*w*cin*kh*kw),'input_shape':tuple(inp[0].shape),'output_shape':tuple(out.shape),'kernel':tuple(m.kernel_size),'stride':tuple(m.stride),'groups':m.groups})
    for name,m in model.named_modules():
        if isinstance(m,nn.Conv2d): hooks.append(m.register_forward_hook(lambda mod,inp,out,n=name:hook(n,mod,inp,out)))
    if hasattr(model,'reset_state'): model.reset_state()
    with torch.no_grad(): model.forward_raw(input_tensor)
    for h in hooks: h.remove()
    return sum(r['macs'] for r in rec),rec

def spike_activity(model):
    spikes=neurons=layers=0; per=[]
    for name,m in model.named_modules():
        if isinstance(m,ParametricLIF):
            spikes+=m.last_spike_count; neurons+=m.last_neuron_count; layers+=1
            per.append({'name':name,'spikes':m.last_spike_count,'neurons':m.last_neuron_count,'rate':m.last_spike_count/m.last_neuron_count if m.last_neuron_count else float('nan')})
    rate=spikes/neurons if neurons else float('nan')
    return {'spike_layers':layers,'spikes':spikes,'spike_neurons_observed':neurons,'spike_rate':rate,'spike_sparsity':1-rate if neurons else float('nan'),'spike_layer_rows':per}

def _prepare_state(model,mode):
    if mode=='reset' and hasattr(model,'reset_state'): model.reset_state()

def benchmark_latency(model,input_tensor,warmup=20,repeats=100,state_mode='stream'):
    """Forward-raw latency. state_mode='stream' preserves SNN state; 'reset' starts each timed forward fresh.

    State reset itself is intentionally outside the timed region, so this compares network execution rather
    than Python bookkeeping overhead.
    """
    if state_mode not in ('stream','reset'): raise ValueError(state_mode)
    device=input_tensor.device; model.eval()
    if hasattr(model,'reset_state'): model.reset_state()
    with torch.no_grad():
        for _ in range(warmup):
            _prepare_state(model,state_mode); model.forward_raw(input_tensor)
    if device.type=='cuda': torch.cuda.synchronize(device)
    times=[]
    with torch.no_grad():
        for _ in range(repeats):
            _prepare_state(model,state_mode)
            if device.type=='cuda':
                a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); model.forward_raw(input_tensor); b.record(); torch.cuda.synchronize(device); times.append(float(a.elapsed_time(b)))
            else:
                t0=time.perf_counter(); model.forward_raw(input_tensor); times.append((time.perf_counter()-t0)*1000)
    s=sorted(times); p95=s[min(len(s)-1,int(round(.95*(len(s)-1))))]; mean=statistics.fmean(times)
    prefix='latency_stream' if state_mode=='stream' else 'latency_reset'
    return {f'{prefix}_mean_ms':mean,f'{prefix}_median_ms':statistics.median(times),f'{prefix}_p95_ms':p95,f'{prefix}_fps':1000/mean if mean>0 else float('inf')}

def peak_cuda_memory(model,input_tensor,state_mode='stream'):
    if input_tensor.device.type!='cuda': return {'peak_vram_mib':float('nan')}
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(input_tensor.device)
    if hasattr(model,'reset_state'): model.reset_state()
    _prepare_state(model,state_mode)
    with torch.no_grad(): model.forward_raw(input_tensor)
    torch.cuda.synchronize(input_tensor.device)
    return {'peak_vram_mib':torch.cuda.max_memory_allocated(input_tensor.device)/2**20}

def benchmark_latency_sequence(model, inputs, warmup=5, repeats=5, state_mode='stream'):
    """Latency over a finite sequence, reporting per-window timing.

    Each repeat traverses the supplied sequence in order. In streaming mode SNN state is
    reset only at the beginning of each repeat; in reset mode it is reset before every window.
    """
    if not inputs: raise ValueError('inputs is empty')
    device=inputs[0].device; model.eval(); times=[]
    def one(x):
        if state_mode=='reset' and hasattr(model,'reset_state'): model.reset_state()
        if device.type=='cuda':
            a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); model.forward_raw(x); b.record(); torch.cuda.synchronize(device); return float(a.elapsed_time(b))
        t0=time.perf_counter(); model.forward_raw(x); return (time.perf_counter()-t0)*1000
    with torch.no_grad():
        for _ in range(warmup):
            if hasattr(model,'reset_state'): model.reset_state()
            for x in inputs: one(x)
        for _ in range(repeats):
            if hasattr(model,'reset_state'): model.reset_state()
            for x in inputs: times.append(one(x))
    s=sorted(times); mean=statistics.fmean(times); p95=s[min(len(s)-1,int(round(.95*(len(s)-1))))]
    prefix='seq_latency_stream' if state_mode=='stream' else 'seq_latency_reset'
    return {f'{prefix}_mean_ms':mean,f'{prefix}_median_ms':statistics.median(times),f'{prefix}_p95_ms':p95,f'{prefix}_fps':1000/mean if mean>0 else float('inf'),f'{prefix}_samples':len(times)}

def spike_activity_sequence(model, inputs, state_mode='stream', dead_rate=1e-5, near_dead_rate=1e-3, saturated_rate=0.80):
    """Measure whole-stream and per-layer spike activity.

    Besides the weighted global activity, aggregate every PLIF layer over all supplied
    windows. ``dead`` and ``saturated`` flags are diagnostics, not accuracy criteria:
    - dead: <= 0.001% activity (default 1e-5 fraction)
    - near-dead: < 0.1% activity
    - saturated: >= 80% activity
    This catches a global average that might otherwise hide pathological layers.
    """
    if hasattr(model,'reset_state'): model.reset_state()
    rows=[]
    layer_totals={}
    with torch.no_grad():
        for i,x in enumerate(inputs):
            if state_mode=='reset' and hasattr(model,'reset_state'): model.reset_state()
            model.forward_raw(x); a=spike_activity(model); rows.append(a)
            for lr in a.get('spike_layer_rows',[]):
                d=layer_totals.setdefault(lr['name'], {'spikes':0,'neurons':0,'window_rates':[]})
                d['spikes'] += int(lr['spikes']); d['neurons'] += int(lr['neurons'])
                if lr['neurons']:
                    d['window_rates'].append(float(lr['rate']))
    if not rows or rows[0]['spike_layers']==0:
        return {'seq_spikes_mean':float('nan'),'seq_spike_rate':float('nan'),'seq_spike_sparsity':float('nan'),'seq_spike_windows':len(rows),'seq_spike_layer_count':0,'seq_spike_dead_layers':0,'seq_spike_near_dead_layers':0,'seq_spike_saturated_layers':0,'seq_spike_health_status':'not_spiking','spike_sequence_rows':rows,'spike_layer_summary_rows':[]}
    spikes=sum(r['spikes'] for r in rows); neurons=sum(r['spike_neurons_observed'] for r in rows); rate=spikes/neurons if neurons else float('nan')
    summary=[]
    for name,d in layer_totals.items():
        r=d['spikes']/d['neurons'] if d['neurons'] else float('nan')
        wr=d['window_rates']
        summary.append({'name':name,'spikes':d['spikes'],'neurons_observed':d['neurons'],'rate':r,'rate_pct':100*r if r==r else float('nan'),'window_rate_mean':statistics.fmean(wr) if wr else float('nan'),'window_rate_min':min(wr) if wr else float('nan'),'window_rate_max':max(wr) if wr else float('nan'),'dead':bool(r==r and r<=dead_rate),'near_dead':bool(r==r and r<near_dead_rate),'saturated':bool(r==r and r>=saturated_rate)})
    rates=[row['rate'] for row in summary if row['rate']==row['rate']]
    dead=sum(r['dead'] for r in summary); near=sum(r['near_dead'] for r in summary); sat=sum(r['saturated'] for r in summary)
    health='ok' if dead==0 and sat==0 else 'warning'
    return {'seq_spikes_mean':spikes/len(rows),'seq_spike_rate':rate,'seq_spike_sparsity':1-rate,'seq_spike_windows':len(rows),'seq_spike_layer_count':len(summary),'seq_spike_layer_rate_min':min(rates) if rates else float('nan'),'seq_spike_layer_rate_median':statistics.median(rates) if rates else float('nan'),'seq_spike_layer_rate_max':max(rates) if rates else float('nan'),'seq_spike_dead_layers':dead,'seq_spike_near_dead_layers':near,'seq_spike_saturated_layers':sat,'seq_spike_health_status':health,'spike_sequence_rows':rows,'spike_layer_summary_rows':summary}
