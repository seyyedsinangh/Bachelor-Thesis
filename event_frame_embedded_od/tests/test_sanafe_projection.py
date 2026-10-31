import pandas as pd

from efod import build_model
from efod.hardware import make_projection_plan, proxy_graph_for_model, write_proxy_architecture_yaml
from efod.models.blocks import PreActSpikingBlock


def _fake_tables(model, windows=2):
    spikes=[]; energies=[]
    for name, module in model.named_modules():
        if isinstance(module, PreActSpikingBlock):
            # 100 output neurons/window, 5% activity.
            spikes.append({
                'name': name+'.act', 'spikes': 10.0,
                'neurons_observed': 200.0, 'rate': 0.05,
            })
            energies.append({
                'window': 0, 'name': name+'.conv', 'kind':'SNN/spiking',
                'theta_in': 10.0, 'synaptic_accs': 300.0,
                'total_energy_pj': 1000.0,
            })
            energies.append({
                'window': 1, 'name': name+'.conv', 'kind':'SNN/spiking',
                'theta_in': 12.0, 'synaptic_accs': 320.0,
                'total_energy_pj': 1100.0,
            })
    # One dense head row/window just to validate the scope split.
    energies += [
        {'window':0,'name':'heads.cls','kind':'ANN/dense','theta_in':0,'synaptic_accs':0,'total_energy_pj':5000.0},
        {'window':1,'name':'heads.cls','kind':'ANN/dense','theta_in':0,'synaptic_accs':0,'total_energy_pj':6000.0},
    ]
    return pd.DataFrame(spikes), pd.DataFrame(energies)


def test_stvgg_projection_plan_and_yaml(tmp_path):
    model=build_model('vgg16','event_snn').eval()
    ss,le=_fake_tables(model)
    plan=make_projection_plan(model,ss,le,windows=2)
    assert len(plan.layers)==11
    assert plan.physical_core_estimate>=11
    assert plan.projected_energy_j_per_window>0
    assert plan.chapter6_dense_heads_uJ>0
    p=write_proxy_architecture_yaml(plan,tmp_path/'proxy.yaml')
    text=p.read_text()
    assert 'architecture:' in text
    assert 'model: leaky_integrate_fire' in text
    assert 'energy_process_spike:' in text


def test_rescat_projection_preserves_branch_joins():
    model=build_model('rescat16','event_snn').eval()
    blocks,edges,joins=proxy_graph_for_model(model)
    assert len(blocks)==14
    assert len(joins)==3
    assert ('backbone.core.stage1.main2','__join_stage1__') in edges
    assert ('backbone.core.stage1.residual','__join_stage1__') in edges
    ss,le=_fake_tables(model)
    plan=make_projection_plan(model,ss,le,windows=2)
    assert len(plan.layers)==14
    assert len(plan.joins)==3
