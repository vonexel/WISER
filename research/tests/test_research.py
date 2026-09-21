import copy
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from research.tools.common import ROOT
from research.tools.prepare_data import component_split
from research.tools.evaluation import metrics, pool, fit_calibration, reports
from research.tools.compare import weighted_auc, paired_bootstrap
from research.tools.spectral import spectrum
from wiser.models.losses import MultiPrototypeHyperspherical, CombinedLoss
from wiser.training.optim import build_optimizer
from wiser.data.awb_sbi import _haar_dwt2d_chw, _haar_idwt2d_chw, awb_self_blend, AWBSBIParams
from wiser.data.sbi import self_blend, SBIParams

def test_component_split_keeps_pair_together():
    ids=[str(i) for i in range(100)];pairs=[(str(i),str(i+1)) for i in range(0,100,2)]
    split,info=component_split(ids,pairs)
    assert all(split[a]==split[b] for a,b in pairs)
    assert split==component_split(ids,pairs)[0]
    assert info['largest_component']==2

def test_component_unknown_id_rejected():
    with pytest.raises(ValueError):component_split(['a'],[('a','b')])

def test_haar_exact_inverse():
    x=np.random.default_rng(0).normal(size=(3,16,16)).astype(np.float32)
    assert np.allclose(_haar_idwt2d_chw(*_haar_dwt2d_chw(x)),x,atol=5e-7)

def test_optimizer_updates_prototypes():
    model=torch.nn.Linear(8,8);loss=MultiPrototypeHyperspherical(embed_dim=8,rp_enabled=True,lambda_real_margin=.5)
    cfg=OmegaConf.create({'lr':.01,'weight_decay':.05,'betas':[.9,.999],'eps':1e-8})
    opt=build_optimizer(model,cfg,loss);before=loss.prototypes.detach().clone()
    loss(model(torch.randn(6,8)),torch.tensor([0,0,0,1,1,1])).backward();opt.step()
    assert not torch.equal(before,loss.prototypes)
    opt.zero_grad(set_to_none=True);assert loss.prototypes.grad is None

def test_pseudos_excluded_from_anchors_and_negative_candidates():
    loss=MultiPrototypeHyperspherical(embed_dim=8,rp_enabled=True,lambda_real_margin=.5)
    x=torch.randn(5,8,requires_grad=True);labels=torch.tensor([0,1,0,1,1]);mask=torch.tensor([1,1,1,0,0])
    a=loss(x,labels,sample_mask=mask);b=loss(x[:3],labels[:3])
    assert torch.allclose(a,b)
    a.backward();assert torch.count_nonzero(x.grad[3:])==0

@pytest.mark.parametrize('labels',[[0,0,0],[1,1,1]])
def test_single_class_loss_is_finite(labels):
    loss=MultiPrototypeHyperspherical(embed_dim=8,rp_enabled=True,lambda_real_margin=.5)
    assert torch.isfinite(loss(torch.randn(3,8),torch.tensor(labels)))

def test_all_soft_loss_is_zero_and_differentiable():
    loss=MultiPrototypeHyperspherical(embed_dim=8)
    x=torch.randn(3,8,requires_grad=True);v=loss(x,torch.tensor([0,1,0]),sample_mask=torch.zeros(3))
    v.backward();assert v.item()==0 and torch.count_nonzero(x.grad)==0

def test_mphc_and_rp_match_paper_equations():
    rng=np.random.default_rng(81)
    z=rng.normal(size=(8,6));z/=np.linalg.norm(z,axis=1,keepdims=True)
    labels=np.array([0,0,0,0,1,1,1,1]);loss=MultiPrototypeHyperspherical(embed_dim=6,rp_enabled=True,lambda_real_margin=.5)
    proto=loss.normalised_prototypes().detach().numpy();s=z@proto.T
    real=s[:,:4].max(1);fake=s[:,4:].max(1)
    pos=np.where(labels==0,real,fake);neg=np.where(labels==0,s[:,4:].mean(1),s[:,:4].mean(1))
    base=np.logaddexp(0,(neg-pos+.35)/.07).mean()
    pair=z@z.T;pair[labels[:,None]==labels[None,:]]=-np.inf
    base+=.1*np.logaddexp(0,(pair.max(1)-pos+.35)/.07).mean()
    w=np.ones(4);w[np.argmin((real-fake)[:4])]=2
    expected=base+.5*np.sum(w*np.maximum(0,(fake-real)[:4]+.4))/w.sum()
    actual=loss(torch.tensor(z,dtype=torch.float32),torch.tensor(labels))
    assert actual.item()==pytest.approx(expected,rel=1e-5)

def test_scientific_loader_rejects_synthetic_status(tmp_path):
    import json
    from research.tools.compare import load_run
    (tmp_path/'status.json').write_text(json.dumps({'status':'complete','scientific_use_allowed':False}))
    with pytest.raises(ValueError,match='scientific'):load_run(tmp_path)

def test_synthetic_manifest_cannot_be_frozen(tmp_path):
    from research.tools.audit_data import audit
    (tmp_path/'SYNTHETIC_ONLY.json').write_text('{}')
    with pytest.raises(ValueError,match='Synthetic'):audit(tmp_path/'frames.csv')

def test_matched_augmentation_rng():
    image=np.random.default_rng(3).integers(0,256,(32,32,3),dtype=np.uint8)
    lm=np.array([[5,5],[27,5],[27,27],[5,27]],dtype=np.float32)
    x=awb_self_blend(image,AWBSBIParams(p=1),np.random.default_rng(44),landmarks=lm)
    y=awb_self_blend(image,AWBSBIParams(p=1),np.random.default_rng(44),landmarks=lm)
    z=self_blend(image,SBIParams(p=1),np.random.default_rng(44),landmarks=lm)
    assert np.array_equal(x[0],y[0]) and np.array_equal(x[2],z[2])

def test_config_one_factor_only():
    base=OmegaConf.to_container(OmegaConf.load(ROOT/'research/configs/E04.yaml'))
    from research.tools.make_configs import VARIANTS
    for name in ['E05','E05b','E06']:
        cfg=OmegaConf.load(ROOT/'research/configs'/f'{name}.yaml');cfg.experiment_id='E04'
        expected=OmegaConf.create(copy.deepcopy(base))
        for key,value in VARIANTS[name].items():OmegaConf.update(expected,key,value)
        assert OmegaConf.to_container(cfg)==OmegaConf.to_container(expected)

def test_video_pool_and_conflicting_labels():
    ids,y,p=pool(np.array([0,0,1]),np.array([.1,.5,.9]),np.array(['a','a','b']))
    assert np.allclose(p,[.3,.9]);assert metrics(y,p)['auc']==1
    with pytest.raises(ValueError):pool(np.array([0,1]),np.array([.2,.3]),np.array(['a','a']))

def test_weighted_auc_ties_matches_explicit_resampling():
    y=np.array([0,0,1,1]);p=np.array([.2,.5,.5,.8]);w=np.array([2,1,3,1])
    assert weighted_auc(y,p,w)==pytest.approx(roc_auc_score(np.repeat(y,w),np.repeat(p,w)))

def test_paired_bootstrap_identical_models_zero():
    y=np.array([0,0,1,1]);a=np.array([[.1,.6,.5,.8]]*3)
    r=paired_bootstrap(y,a,a,20)
    assert r['mean_difference']==[0,0] and np.all(np.array(r['hierarchical_95_ci'])==0)

def test_spectral_power_tracks_amplitude():
    yy,xx=np.mgrid[:256,:256]
    a=np.repeat((128+10*np.sin(xx*2*np.pi/8))[...,None],3,axis=2)
    b=np.repeat((128+20*np.sin(xx*2*np.pi/8))[...,None],3,axis=2)
    _,ea=spectrum(a);_,eb=spectrum(b)
    assert eb.sum()==pytest.approx(ea.sum()*4,rel=1e-8)

def test_calibration_fit_is_source_only_and_monotone():
    z=np.array([-2,-1,-.5,.2,.3,.5,1,2],dtype=float);y=np.array([0,0,1,0,1,0,1,1]);ids=np.array([str(i) for i in range(8)])
    c=fit_calibration(z,y,ids);assert c['temperature']>0
    r=reports({'logits':z,'labels':y,'video_ids':np.array(['real/'+str(i) if v==0 else 'fake/'+str(i) for i,v in enumerate(y)])},c)
    assert r['raw_fixed']['frame']['auc']==r['calibrated']['frame']['auc']
