import json
import xml.etree.ElementTree as ET
import numpy as np
from research.tools.common import write_json
from research.tools.evaluation import reports
from research.tools.plot_results import plot_run, plot_comparisons

def test_run_svgs_and_numeric_sources(tmp_path):
    run=tmp_path/'run';run.mkdir()
    write_json(run/'passport.json',{'scientific_use_allowed':False})
    c={'temperature':1.,'bias':.1,'raw_threshold':.5,'calibrated_threshold':.5}
    write_json(run/'calibration.json',c)
    write_json(run/'epoch_history.json',[{'epoch':1,'train_loss':.6,'val_loss':.7,'val_auc_ema':.75,'lr':.001}])
    d={'logits':np.array([-1.,-.5,.3,1.]),'labels':np.array([0,0,1,1]),
       'video_ids':np.array(['real/a','real/a','fake/b','fake/b'])}
    for scope in ['val','test','external']:
        np.savez_compressed(run/f'{scope}_predictions.npz',**d)
        write_json(run/f'{scope}_metrics.json',reports(d,c))
    out=plot_run(run)
    paths=list(out.glob('*.svg'));assert len(paths)==10
    for p in paths:
        assert ET.parse(p).getroot().tag.endswith('svg')
        assert 'NOT SCIENTIFIC EVIDENCE' in p.read_text(encoding='utf-8')
    assert json.loads((out/'sources.json').read_text())['scientific_use_allowed'] is False
    assert (out/'plot_data.json').exists()

def test_comparison_svgs_allow_point_outside_percentile_interval(tmp_path):
    # Percentile intervals need not contain the sample point estimate.
    value={'scientific_use_allowed':False,'contrasts':{'E04-E05':{
        'mean_difference':[0.,0.], 'hierarchical_familywise_95_ci':[[.01,.03],[-.03,-.01]]}},
        'summaries':{name:{k:{'mean':.6,'sd':.01} for k in ['auc','RR','FR','bAcc','Brier','ECE']}
                     for name in ['E04','E05','E05b','E06']}}
    path=tmp_path/'comparisons.json';write_json(path,value)
    out=plot_comparisons(path)
    assert len(list(out.glob('*.svg')))==2
    for p in out.glob('*.svg'):ET.parse(p)
