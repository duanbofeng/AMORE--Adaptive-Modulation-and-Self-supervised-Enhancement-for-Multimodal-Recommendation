#!/usr/bin/env python3
"""Reproducible illustrative scenarios, independent of all measured repo results."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SEED = 261109
RNG = np.random.default_rng(SEED)
SEEDS = [17, 43, 89, 131, 211]
DATASETS = ['Baby', 'Sports', 'Clothing']
MODELS = ['LightGCN', 'MGCN', 'GUME', 'MENTOR', 'AMORE']
METRICS = ['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20']
COLORS = {'LightGCN': '#88939b', 'MGCN': '#9b83a6', 'GUME': '#387b9c',
          'MENTOR': '#69a389', 'AMORE': '#c47749'}
FIGURES = []


def export(frame, name):
    frame = frame.copy()
    frame.insert(0, 'provenance', 'synthetic_learning_example')
    frame.to_csv(ROOT / 'data' / (name + '.csv'), index=False, float_format='%.10g')
    return frame


def summarize(frame, groups, metrics):
    rows = []
    for key, group in frame.groupby(groups, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        for metric in metrics:
            v = group[metric].to_numpy()
            n = len(v); mean = v.mean(); sd = v.std(ddof=1)
            half = stats.t.ppf(.975, n-1)*sd/np.sqrt(n)
            rows.append(dict(zip(groups, key), metric=metric, n=n, mean=mean,
                             sd=sd, ci95_low=mean-half, ci95_high=mean+half))
    return pd.DataFrame(rows)


def save(fig, stem, title, caption):
    fig.suptitle(title, fontsize=13, y=.995)
    fig.text(.99, .005, 'Illustrative simulation · five synthetic runs', ha='right',
             va='bottom', fontsize=7, color='#666666')
    fig.tight_layout(rect=(0, .035, 1, .95))
    for ext in ['png', 'svg']:
        path=ROOT/'figures'/(stem+'.'+ext)
        fig.savefig(path, dpi=220, facecolor='white')
        if ext=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    FIGURES.append(dict(stem=stem, title=title, caption=caption))


def build_performance():
    # Independent scenario parameters, NOT copied or fitted to experimental files.
    anchors = {'Baby': [.0643, .1027, .0339, .0446],
               'Sports': [.0748, .1189, .0398, .0524],
               'Clothing': [.0441, .0703, .0228, .0299]}
    # Improvements are relative percentages, with a deliberately mixed Clothing outcome.
    offsets = {'LightGCN': [-.115, -.102, -.121], 'MGCN': [-.017, -.012, -.009],
               'GUME': [0., 0., 0.], 'MENTOR': [.003, .0045, .006], 'AMORE': [.012, .009, .004]}
    rows = []
    for di, ds in enumerate(DATASETS):
        base = np.array(anchors[ds])
        shared = RNG.normal(0, .008, (5, 1))
        for model in MODELS:
            systematic = np.repeat(offsets[model][di], 4)
            systematic += RNG.normal(0, .0018, 4)
            if model == 'AMORE' and ds == 'Clothing':
                systematic[3] = .001
            residual = RNG.normal(0, .004, (5, 1)) + RNG.normal(0, .0025, (5, 4))
            values = base*(1 + systematic + shared + residual)
            for seed, value in zip(SEEDS, values):
                rows.append(dict(dataset=ds, model=model, seed=seed, **dict(zip(METRICS, value))))
    frame = pd.DataFrame(rows)
    export(frame, 'performance_runs')
    summary = summarize(frame, ['dataset', 'model'], METRICS)
    export(summary, 'table_01_performance')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for col, ds in enumerate(DATASETS):
        for ax, metric in zip(axes[:, col], ['recall@20', 'ndcg@20']):
            sub = frame[frame.dataset == ds]
            for x, model in enumerate(MODELS):
                v = sub[sub.model == model][metric]
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), marker='o', capsize=4,
                            color=COLORS[model], markersize=6)
                ax.scatter(x + np.linspace(-.10, .10, 5), v, s=12, alpha=.45, color=COLORS[model])
            ax.set(xticks=range(5), xticklabels=MODELS, title=ds, ylabel=metric)
            ax.tick_params(axis='x', rotation=25)
    save(fig, '01_performance', 'Overall performance with run-to-run variation',
         'Points show means and error bars show sample SD across five synthetic runs; small dots retain individual runs. Model and dataset names identify illustrative roles, not measured benchmark results.')
    return frame


def build_ablations(perf):
    # Additive changes relative to the full model, scaled to each dataset.
    effects = {
        'Neither branch': [-.00085, -.00014, .00020],
        'Graph only': [-.00039, -.00023, -.00008],
        'Fusion only': [-.00056, .00004, .00012],
        'Full': [0., 0., 0.],
        'Shuffled prior': [-.00037, -.00008, -.00003],
        'Constant prior': [-.00022, .00003, .00005],
        'Detached prior': [-.00004, -.00002, .00001],
        'No moments': [-.00045, -.00024, .00006],
        'No InfoNCE': [-.0049, -.0041, -.0033],
        'No moments or InfoNCE': [-.0052, -.0045, -.0032],
        'No BM': [-.00115, -.00082, -.00065],
        'No extended BM': [-.00055, -.00035, -.00018],
        'No UM cross': [-.00138, -.00104, -.00059],
        'No UM noise': [-.00042, -.00016, .00016],
        'No representation L2': [-.00017, -.00019, .00002],
        'No extended L2': [-.00026, -.00012, -.00009],
    }
    rows = []
    for di, ds in enumerate(DATASETS):
        reference = perf[(perf.dataset == ds) & (perf.model == 'AMORE')].set_index('seed')
        for name, changes in effects.items():
            for seed in SEEDS:
                delta = changes[di] + (RNG.normal(0, .00036) if name != 'Full' else 0)
                values = reference.loc[seed, METRICS].to_numpy(dtype=float).copy()
                values += delta*np.array([.66, 1., .32, .43])
                if name != 'Full':
                    values += RNG.normal(0, .000055, 4)
                rows.append(dict(dataset=ds, variant=name, seed=seed, **dict(zip(METRICS, values))))
    frame = pd.DataFrame(rows); export(frame, 'ablation_runs')
    paired = []
    for ds in DATASETS:
        for metric in METRICS:
            table = frame[frame.dataset == ds].pivot(index='seed', columns='variant', values=metric)
            for variant in effects:
                if variant == 'Full':
                    continue
                v = table[variant]-table.Full
                half = stats.t.ppf(.975, 4)*v.std(ddof=1)/np.sqrt(5)
                paired.append(dict(dataset=ds, variant=variant, metric=metric, n=5,
                                   delta=v.mean(), ci95_low=v.mean()-half, ci95_high=v.mean()+half,
                                   relative_percent=100*v.mean()/table.Full.mean()))
    export(pd.DataFrame(paired), 'table_02_ablation_effects')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, ds in zip(axes, DATASETS):
        group = frame[frame.dataset == ds]
        for label, variants, color in [('Graph off', ['Neither branch', 'Fusion only'], '#387b9c'),
                                       ('Graph on', ['Graph only', 'Full'], '#c47749')]:
            values = [group[group.variant == v]['recall@20'] for v in variants]
            ax.errorbar([0, 1], [v.mean() for v in values], yerr=[v.std(ddof=1) for v in values],
                        marker='o', capsize=4, label=label, color=color)
        ax.set(xticks=[0,1], xticklabels=['Fusion off','Fusion on'], ylabel='Recall@20', title=ds)
        ax.legend(fontsize=8)
    save(fig, '02_branch_factorial', 'Where affinity helps: a mixed factorial scenario',
         'The same backbone is represented with graph and fusion switches. Error bars are sample SD. Stronger total performance does not by itself establish a positive interaction between branches.')
    names = ['No moments', 'No BM', 'No extended BM', 'No UM cross', 'No UM noise',
             'No representation L2', 'No extended L2']
    effect = pd.DataFrame(paired)
    matrix = effect[(effect.metric == 'recall@20') & effect.variant.isin(names)].pivot(
        index='variant', columns='dataset', values='relative_percent').loc[names, DATASETS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios':[1.25,1]})
    limit = max(abs(matrix.to_numpy()).max(), 1.)
    heat = axes[0].imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    axes[0].set(xticks=range(3), xticklabels=DATASETS, yticks=range(len(names)), yticklabels=names,
                title='Removal - full: Recall@20 change (%)')
    for i in range(len(names)):
        for j in range(3):
            axes[0].text(j, i, '{:+.2f}'.format(matrix.iloc[i,j]), ha='center', va='center', fontsize=9,
                         color='white' if abs(matrix.iloc[i,j])>limit*.65 else '#222')
    fig.colorbar(heat, ax=axes[0], fraction=.045, pad=.025)
    sub = frame[frame.dataset == 'Baby']
    for label, variants in [('InfoNCE on', ['No moments','Full']),
                            ('InfoNCE off', ['No moments or InfoNCE','No InfoNCE'])]:
        v = [sub[sub.variant == name]['recall@20'] for name in variants]
        axes[1].errorbar([0,1], [x.mean() for x in v], yerr=[x.std(ddof=1) for x in v], marker='o', capsize=4, label=label)
    axes[1].set(xticks=[0,1], xticklabels=['Moments off','Moments on'], ylabel='Recall@20', title='Baby: moments x InfoNCE')
    axes[1].legend()
    save(fig, '03_loss_analysis', 'Auxiliary objectives: useful, weak and occasionally redundant',
         'The heatmap reports relative changes after removing one loss. Blue means removal hurts performance; red means removal improves it. The factorial panel keeps both representation penalties fixed. Heatmap colors are effect sizes, not significance labels.')
    return frame


def build_sensitivity(perf, ablations):
    grids = {'Moment weight':[0,.01,.03,.1,.3,1],
             'Representation L2':[0,1e-6,3e-6,1e-5,3e-5,1e-4],
             'Extended L2':[0,.001,.01,.1,.3,1]}
    penalties = {'Moment weight':[-.0004,-.00015,.00012,0,-.00035,-.0015],
                 'Representation L2':[-.00022,-.00014,-.00005,0,-.00010,-.0008],
                 'Extended L2':[-.0003,-.00024,-.00010,0,-.00042,-.0016]}
    rows=[]
    zero_variants={'Moment weight':'No moments', 'Representation L2':'No representation L2',
                   'Extended L2':'No extended L2'}
    for di, ds in enumerate(DATASETS):
        base=perf[(perf.dataset==ds)&(perf.model=='AMORE')].set_index('seed')
        for name, grid in grids.items():
            for j, weight in enumerate(grid):
                for seed in SEEDS:
                    delta=penalties[name][j]*(1-.18*di)+RNG.normal(0,.00020)
                    if j==3:
                        delta=0.
                    recall=base.loc[seed,'recall@20']+delta
                    ndcg=base.loc[seed,'ndcg@20']+.43*delta+(RNG.normal(0,.000045) if j!=3 else 0)
                    if weight==0:
                        zero=ablations[(ablations.dataset==ds)&(ablations.seed==seed)&
                                       (ablations.variant==zero_variants[name])].iloc[0]
                        recall,ndcg=zero['recall@20'],zero['ndcg@20']
                    rows.append(dict(dataset=ds, parameter=name, weight=weight, seed=seed,
                                     recall=recall, ndcg=ndcg))
    frame=pd.DataFrame(rows);export(frame,'sensitivity_runs')
    fig,axes=plt.subplots(3,3,figsize=(12,8.4))
    for i,ds in enumerate(DATASETS):
        for j,name in enumerate(grids):
            s=frame[(frame.dataset==ds)&(frame.parameter==name)].groupby('weight').recall.agg(['mean','std'])
            x=np.arange(len(s));ax=axes[i,j]
            ax.plot(x,s['mean'],marker='o',color=COLORS['AMORE'])
            ax.fill_between(x,s['mean']-s['std'],s['mean']+s['std'],color=COLORS['AMORE'],alpha=.14)
            ax.set(xticks=x,xticklabels=['{:g}'.format(v) for v in s.index],title=name if i==0 else '',
                   ylabel=ds+' Recall@20' if j==0 else '')
            ax.tick_params(axis='x',labelsize=8)
    save(fig,'04_sensitivity','Broad plateaus and deteriorating extremes',
         'One coefficient changes at a time. Bands show sample SD. Zero and positive values are displayed at categorical spacing; the plotted high point is not a new configuration-selection recommendation.')


def build_robustness(perf):
    rows=[];rates=[0,.05,.10,.20,.30]
    for ds in DATASETS:
        for kind, scale in [('Interaction noise',.30),('Image mismatch',.20),('Text mismatch',.41)]:
            for model in ['GUME','MENTOR','AMORE']:
                base=perf[(perf.dataset==ds)&(perf.model==model)].set_index('seed')
                resilience={'GUME':1.,'MENTOR':.92,'AMORE':.81}[model]
                if ds=='Clothing' and kind=='Text mismatch' and model=='AMORE':
                    resilience=1.03
                run_slopes=RNG.normal(1,.11,5)
                for rate in rates:
                    for si,seed in enumerate(SEEDS):
                        drop=scale*(rate+1.5*rate**2)*resilience*run_slopes[si]
                        for metric in ['recall@20','ndcg@20']:
                            clean=base.loc[seed,metric]
                            val=clean*(1-drop)+(RNG.normal(0,clean*.0022) if rate else 0)
                            rows.append(dict(dataset=ds,kind=kind,model=model,seed=seed,rate=rate,
                                             metric=metric,value=val,clean=clean,degradation_percent=100*(1-val/clean)))
    frame=pd.DataFrame(rows);export(frame,'robustness_runs')
    for ds in DATASETS:
        fig,axes=plt.subplots(2,3,figsize=(12,6.8))
        for j,kind in enumerate(frame.kind.unique()):
            for model in ['GUME','MENTOR','AMORE']:
                sub=frame[(frame.dataset==ds)&(frame.kind==kind)&(frame.model==model)&(frame.metric=='recall@20')]
                for ax,metric in [(axes[0,j],'value'),(axes[1,j],'degradation_percent')]:
                    s=sub.groupby('rate')[metric].agg(['mean','std'])
                    ax.plot(s.index,s['mean'],marker='o',label=model,color=COLORS[model],markersize=4)
                    ax.fill_between(s.index,s['mean']-s['std'],s['mean']+s['std'],color=COLORS[model],alpha=.12)
            axes[0,j].set(title=kind,ylabel='Recall@20');axes[0,j].legend(fontsize=8)
            axes[1,j].set(xlabel='Selected corruption rate',ylabel='Drop from own clean score (%)')
        save(fig,'05_robustness_'+ds.lower(),ds+': absolute performance and corruption sensitivity',
             'Lines summarize synthetic retraining scenarios under each corruption level; bands are sample SD. Lower degradation is preferable, but must be interpreted alongside the absolute clean score. Clothing deliberately includes a text-noise failure case.')


def build_cohorts(perf):
    rows=[]
    for ds in DATASETS:
        for model in ['GUME','MENTOR','AMORE']:
            base=perf[(perf.dataset==ds)&(perf.model==model)].set_index('seed')
            for seed in SEEDS:
                # User groups form a partition; weighted group mean equals the overall value.
                weights=np.array([.36,.33,.31]);factors=np.array([.77,1.01,1.25])
                factors+=RNG.normal(0,.012,3)
                if model=='AMORE':
                    factors+=np.array([.012,0,-.009])
                factors/=weights@factors
                for group,w,f in zip(['Sparse','Medium','Dense'],weights,factors):
                    rows.append(dict(dataset=ds,model=model,seed=seed,family='User activity',group=group,
                                     weight=w,recall=base.loc[seed,'recall@20']*f,ndcg=base.loc[seed,'ndcg@20']*f))
                for qi,group in enumerate(['Tail Q1','Q2','Q3','Q4','Head Q5']):
                    f=[.46,.65,.84,1.05,1.35][qi]+RNG.normal(0,.012)
                    if model=='AMORE':
                        f+=[.009,.006,.004,-.003,-.006][qi]
                    rows.append(dict(dataset=ds,model=model,seed=seed,family='Item degree',group=group,
                                     weight=np.nan,recall=base.loc[seed,'recall@20']*f,ndcg=base.loc[seed,'ndcg@20']*f))
                f={'GUME':.254,'MENTOR':.267,'AMORE':.273}[model]+RNG.normal(0,.012)
                if ds=='Clothing' and model=='AMORE':
                    f-=.013
                rows.append(dict(dataset=ds,model=model,seed=seed,family='Zero-interaction scenario',group='Cold items',
                                 weight=np.nan,recall=base.loc[seed,'recall@20']*f,ndcg=base.loc[seed,'ndcg@20']*f))
    frame=pd.DataFrame(rows);export(frame,'cohort_runs')
    export(summarize(frame,['dataset','family','group','model'],['recall','ndcg']),'table_03_cohorts')
    for ds in DATASETS:
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),gridspec_kw={'width_ratios':[1,1.3,.8]})
        for ax,family,order in zip(axes,['User activity','Item degree','Zero-interaction scenario'],
                                  [['Sparse','Medium','Dense'],['Tail Q1','Q2','Q3','Q4','Head Q5'],['Cold items']]):
            for mi,model in enumerate(['GUME','MENTOR','AMORE']):
                sub=frame[(frame.dataset==ds)&(frame.family==family)&(frame.model==model)]
                s=sub.groupby('group').recall.agg(['mean','std']).loc[order]
                x=np.arange(len(order))+(mi-1)*.08
                ax.errorbar(x,s['mean'],yerr=s['std'],marker='o',capsize=3,label=model,color=COLORS[model],markersize=4)
            ax.set(xticks=range(len(order)),xticklabels=order,title=family,ylabel='Recall@20')
            ax.tick_params(axis='x',labelsize=8)
        axes[0].legend(fontsize=8)
        save(fig,'06_cohorts_'+ds.lower(),ds+': separate activity, popularity and cold-item views',
             'User activity groups form a weighted partition. Item groups use conditional group metrics and need not sum to the overall score. The cold-item panel is a separate zero-training-interaction scenario, not a synonym for warm long-tail items.')


def build_efficiency(perf):
    assumptions={'LightGCN':[2.28,.78,480,.16], 'MGCN':[34.7,1.68,970,.24],
                 'GUME':[35.6,2.06,1190,.29], 'MENTOR':[37.1,2.34,1450,.35],
                 'AMORE':[36.3,2.31,1330,.33]}
    rows=[]
    for model, (params,epoch,memory,inference) in assumptions.items():
        for seed in SEEDS:
            rows.append(dict(model=model,seed=seed,parameters_million=params,
                             epoch_seconds=epoch*RNG.lognormal(-.002,.052),
                             peak_mib=memory*RNG.lognormal(0,.009),
                             inference_seconds=inference*RNG.lognormal(-.002,.043)))
    frame=pd.DataFrame(rows);export(frame,'efficiency_runs')
    export(summarize(frame,['model'],['parameters_million','epoch_seconds','peak_mib','inference_seconds']),'table_04_efficiency')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for model in MODELS:
        x=frame[frame.model==model];y=perf[(perf.dataset=='Baby')&(perf.model==model)]['recall@20']
        axes[0].errorbar(x.epoch_seconds.mean(),y.mean(),xerr=x.epoch_seconds.std(),yerr=y.std(),fmt='none',color=COLORS[model],alpha=.7)
        axes[0].scatter(x.epoch_seconds.mean(),y.mean(),s=x.peak_mib.mean()/8,color=COLORS[model],alpha=.8)
        axes[0].annotate(model,(x.epoch_seconds.mean(),y.mean()),xytext=(6,5),textcoords='offset points',fontsize=8)
    axes[0].set(xlabel='Assumed seconds / epoch',ylabel='Baby Recall@20',title='Accuracy-cost trade-off; area proportional to MiB')
    grouped=frame.groupby('model').inference_seconds.agg(['mean','std']).loc[MODELS]
    axes[1].barh(MODELS,grouped['mean'],xerr=grouped['std'],color=[COLORS[m] for m in MODELS],height=.6,capsize=3)
    axes[1].set(xlabel='Assumed seconds / full ranking pass',title='Inference cost')
    save(fig,'07_efficiency','A modest gain can still carry a computational cost',
         'Timing and memory values are hypothetical assumptions, not hardware measurements. The proposed model is not designed to win both accuracy and efficiency in this example. Parameter counts are fixed across runs; timing variation is not parameter variation.')


def build_convergence(perf):
    rows=[];epochs=np.arange(1,101)
    for model in ['GUME','MENTOR','AMORE']:
        base=perf[(perf.dataset=='Baby')&(perf.model==model)].set_index('seed')
        for seed in SEEDS:
            target=base.loc[seed,'recall@20']*1.018
            speed={'GUME':13.8,'MENTOR':15.3,'AMORE':16.1}[model]*RNG.uniform(.95,1.05)
            curve=.021+(target-.021)*(1-np.exp(-epochs/speed))
            curve-=.0030*np.exp(-((epochs-22)/5.5)**2)
            curve-=np.maximum(epochs-70,0)*.000045
            curve+=np.convolve(RNG.normal(0,.00075,104),np.ones(5)/5,'valid')
            for ep,value in zip(epochs,curve):
                rows.append(dict(model=model,seed=seed,epoch=ep,validation_recall=value,
                                 training_bpr=.66*np.exp(-ep/24)+.041+RNG.normal(0,.0017)))
    frame=pd.DataFrame(rows);export(frame,'convergence_runs')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for model in ['GUME','MENTOR','AMORE']:
        for ax,metric in zip(axes,['validation_recall','training_bpr']):
            s=frame[frame.model==model].groupby('epoch')[metric].agg(['mean','std'])
            ax.plot(s.index,s['mean'],label=model,color=COLORS[model])
            ax.fill_between(s.index,s['mean']-s['std'],s['mean']+s['std'],color=COLORS[model],alpha=.15)
    axes[0].set(xlabel='Epoch',ylabel='Validation Recall@20',title='Plateau, local dip and late decline')
    axes[1].set(xlabel='Epoch',ylabel='BPR objective',title='Training loss is not validation quality')
    for ax in axes: ax.legend(fontsize=8)
    save(fig,'08_convergence','Training trajectories need not improve monotonically',
         'These are generated validation trajectories and objective traces, separate from the simulated holdout table. Bands show run SD; a lower training loss does not establish better generalization or faster convergence.')


def build_interpretations():
    cases={'Consistent gain':(.00105,.00022), 'Small uncertain gain':(.00022,.00067),
           'Heterogeneous response':(.00035,.0011), 'Consistent degradation':(-.00078,.00018)}
    rows=[];fig,ax=plt.subplots(figsize=(9,3.7))
    for i,(name,(mean,scale)) in enumerate(cases.items()):
        values=RNG.normal(mean,scale,5);sd=values.std(ddof=1);half=stats.t.ppf(.975,4)*sd/np.sqrt(5)
        ax.errorbar(values.mean()*1000,i,xerr=half*1000,fmt='o',capsize=4,color='#345c78')
        ax.scatter(values*1000,i+np.linspace(-.1,.1,5),color='#c47749',s=22)
        for seed,value in zip(SEEDS,values): rows.append(dict(scenario=name,seed=seed,paired_delta=value))
    ax.axvline(0,color='#444',ls='--',lw=.8)
    ax.set(yticks=range(4),yticklabels=list(cases),xlabel='Paired Recall@20 difference (x 1,000)');ax.invert_yaxis()
    export(pd.DataFrame(rows),'interpretation_runs')
    save(fig,'09_interpretation','Four kinds of evidence deserve four different conclusions',
         'Intervals are paired t intervals from five generated run differences. A positive mean with an interval crossing zero is inconclusive; lack of significance is not proof of equivalence. The scenario names describe intended generating settings, not guaranteed statistical outcomes.')


def main():
    for directory in ['data','figures']:
        (ROOT/directory).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.13,
                         'axes.axisbelow':True,'svg.fonttype':'none'})
    perf=build_performance();ablations=build_ablations(perf)
    build_sensitivity(perf,ablations);build_robustness(perf);build_cohorts(perf)
    build_efficiency(perf);build_convergence(perf);build_interpretations()
    # Meaningful constraints, rather than rounding metrics into an attractive story.
    for frame in [perf,ablations]:
        assert frame[METRICS].gt(0).all().all() and frame[METRICS].lt(1).all().all()
        assert (frame['recall@20'] >= frame['recall@10']).all()
        assert (frame['ndcg@20'] >= frame['ndcg@10']).all()
        assert (frame['ndcg@20'] <= frame['recall@20']).all()
    manifest=dict(provenance='synthetic_learning_example', generator_seed=SEED, synthetic_run_ids=SEEDS,
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source='Scenario assumptions and NumPy draws only; no repository experiment files read.',
                  interpretation='Illustrations of evidence patterns, not expected results or claims about named methods.',
                  dependency_versions={'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
                  figures=FIGURES)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Generated',len(FIGURES),'figures and',len(list((ROOT/'data').glob('*.csv'))),'data tables in',ROOT)


if __name__=='__main__':
    main()
