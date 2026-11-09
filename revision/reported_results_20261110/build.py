#!/usr/bin/env python3
"""Plot only the author's supplied results; no random draws or simulated imports."""
import base64
import hashlib
import html
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SOURCE = REPO / 'papers/AMORE.docx'
INPUT = json.loads((ROOT/'inputs.json').read_text())
DATASETS = INPUT['datasets']
METRICS = INPUT['metrics']
COLORS = ['#75938c', '#91839d', '#76889b', '#287b9b', '#c36a42']
FIGURES = []


def export(frame, name):
    frame.to_csv(ROOT/'data'/(name+'.csv'), index=False, float_format='%.10g')


def save(fig, name, caption):
    fig.tight_layout(pad=1.3)
    for ext in ['png', 'svg', 'pdf']:
        path=ROOT/'figures'/(name+'.'+ext)
        fig.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        if ext=='svg':
            path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
    plt.close(fig)
    FIGURES.append(dict(name=name, caption=caption))


def tables():
    perf=pd.DataFrame([dict(model=label.split(' (')[0],label=label,dataset=ds,
                           **dict(zip(METRICS,values[di*4:di*4+4])))
                       for label,values in INPUT['performance'].items() for di,ds in enumerate(DATASETS)])
    ab=pd.DataFrame([dict(variant=name,dataset=ds,**dict(zip(INPUT['ablation_metrics'],values[di])))
                     for name,values in INPUT['ablation'].items() for di,ds in enumerate(DATASETS)])
    factor=pd.DataFrame([dict(variant=variant,dataset=ds,propagation=int(variant[0]),fusion=int(variant[1]),
                             metric=INPUT['factorial_metric'],value=values[di])
                         for variant,values in INPUT['factorial'].items() for di,ds in enumerate(DATASETS)])
    gains=[];drops=[];interactions=[]
    for ds in DATASETS:
        sub=perf[perf.dataset==ds].set_index('model')
        baseline=sub.drop('AMORE')
        for metric in METRICS:
            best=baseline[metric].idxmax();base=baseline.loc[best,metric];value=sub.loc['AMORE',metric]
            gains.append(dict(dataset=ds,metric=metric,comparator=best,baseline=base,amore=value,
                              absolute_difference=value-base,relative_percent=100*(value/base-1)))
        full=ab[(ab.dataset==ds)&(ab.variant=='Full')].iloc[0]
        for row in ab[(ab.dataset==ds)&(ab.variant!='Full')].to_dict('records'):
            for metric in INPUT['ablation_metrics']:
                drops.append(dict(dataset=ds,variant=row['variant'],metric=metric,full=full[metric],
                                  ablated=row[metric],absolute_drop=full[metric]-row[metric],
                                  drop_percent=100*(1-row[metric]/full[metric])))
        y=factor[factor.dataset==ds].set_index('variant').value
        interactions.append(dict(dataset=ds,metric=INPUT['factorial_metric'],y00=y['00'],y10=y['10'],y01=y['01'],y11=y['11'],
                                 propagation_only=y['10']-y['00'],fusion_only=y['01']-y['00'],
                                 joint_gain=y['11']-y['00'],interaction=y['11']-y['10']-y['01']+y['00'],
                                 propagation_given_fusion=y['11']-y['01'],fusion_given_propagation=y['11']-y['10'],
                                 full_minus_best_single=y['11']-max(y['10'],y['01']),
                                 joint_relative_percent=100*(y['11']/y['00']-1)))
    gains=pd.DataFrame(gains);drops=pd.DataFrame(drops);interactions=pd.DataFrame(interactions)
    for frame,name in [(perf,'performance'),(ab,'ablation'),(factor,'factorial'),(gains,'performance_gains'),
                       (drops,'ablation_drops'),(interactions,'factorial_effects')]: export(frame,name)
    return perf,ab,factor,gains,drops,interactions


def plots(perf,ab,factor,gains,drops,effects):
    names=['FREEDOM','MGCN','DHMRec','MENTOR','AMORE']
    fig,axes=plt.subplots(2,3,figsize=(12,6.4))
    for j,ds in enumerate(DATASETS):
        sub=perf[perf.dataset==ds].set_index('model')
        for ax,metric in zip(axes[:,j],['Recall@20','NDCG@20']):
            vals=sub.loc[names,metric].to_numpy()
            for x,(value,color) in enumerate(zip(vals,COLORS)):
                ax.scatter(x,value,color=color,s=43,zorder=3)
                ax.annotate(f'{value:.4f}',(x,value),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            ax.set(xticks=range(5),xticklabels=names,title=ds,ylabel=metric)
            ax.tick_params(axis='x',rotation=25,labelsize=8)
            ax.margins(y=.3,x=.15)
    save(fig,'01_overall_comparison','Recall@20 and NDCG@20 for five selected multimodal methods on Baby, Sports, and Clothing. Points show the reported values. The complete comparison includes twelve baselines in Table 2. No uncertainty estimates are available from the supplied aggregate results.')

    fig,ax=plt.subplots(figsize=(8.2,3.4))
    vals=gains.pivot(index='dataset',columns='metric',values='relative_percent').loc[DATASETS,METRICS]
    im=ax.imshow(vals,cmap='RdBu',vmin=-4,vmax=4,aspect='auto')
    for i in range(3):
        for j in range(4): ax.text(j,i,f'{vals.iloc[i,j]:+.2f}%',ha='center',va='center',color='white' if abs(vals.iloc[i,j])>2.4 else '#202b31',fontsize=12)
    ax.set(xticks=range(4),xticklabels=METRICS,yticks=range(3),yticklabels=DATASETS,title='AMORE relative to the strongest reported baseline')
    ax.grid(False);fig.colorbar(im,ax=ax,label='Relative change (%)',fraction=.04,pad=.04)
    save(fig,'02_relative_improvement','Relative change of AMORE against the strongest baseline for each dataset and metric: 100 × (AMORE − baseline) / baseline. MENTOR is the strongest baseline in all twelve comparisons. The negative Baby Recall@20 entry is retained.')

    variants=['Full','w/o ELAM','w/o CFDM','w/o MACA']
    fig,axes=plt.subplots(2,3,figsize=(12,6.4))
    for j,ds in enumerate(DATASETS):
        sub=ab[ab.dataset==ds].set_index('variant')
        for ax,metric in zip(axes[:,j],INPUT['ablation_metrics']):
            vals=sub.loc[variants,metric]
            ax.bar(range(4),vals,color=['#c36a42','#7197a8','#91aaa1','#93839c'],width=.62)
            for x,value in enumerate(vals): ax.text(x,value+vals.max()*.025,f'{value:.4f}',ha='center',fontsize=9)
            ax.set(xticks=range(4),xticklabels=['Full','w/o\nELAM','w/o\nCFDM','w/o\nMACA'],title=ds,ylabel=metric,ylim=(0,vals.max()*1.18))
            ax.grid(axis='x',visible=False)
    save(fig,'03_module_ablation','Module-removal results on the three datasets. The full model has the highest reported Recall@20 and NDCG@20 among the four variants. MACA removal produces the largest decrease in each comparison; this aggregate removal does not isolate its individual loss terms.')

    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    vmax=np.ceil(drops.drop_percent.max())
    for ax,metric in zip(axes,INPUT['ablation_metrics']):
        vals=drops[drops.metric==metric].pivot(index='variant',columns='dataset',values='drop_percent').loc[variants[1:],DATASETS]
        im=ax.imshow(vals,cmap='Blues',vmin=0,vmax=vmax,aspect='auto')
        for i in range(3):
            for j in range(3): ax.text(j,i,f'{vals.iloc[i,j]:.2f}%',ha='center',va='center',fontsize=11,color='white' if vals.iloc[i,j]>vmax*.55 else '#20333f')
        ax.set(xticks=range(3),xticklabels=DATASETS,yticks=range(3),yticklabels=variants[1:],title=metric);ax.grid(False)
        fig.colorbar(im,ax=ax,label='Decrease from full (%)',fraction=.04,pad=.04)
    save(fig,'04_module_drop','Relative performance decrease after removing each module, calculated as 100 × (Full − ablation) / Full. The common color scale shows effect magnitude, not statistical significance. ELAM and CFDM have metric-dependent contributions.')

    fig,axes=plt.subplots(1,3,figsize=(11.8,3.8))
    for ax,ds in zip(axes,DATASETS):
        y=factor[factor.dataset==ds].set_index('variant').value
        for propagation,color in [(0,'#387b9c'),(1,'#c36a42')]:
            vals=[y[f'{propagation}{fusion}'] for fusion in [0,1]]
            ax.plot([0,1],vals,marker='o',color=color,label='Propagation '+('on' if propagation else 'off'))
            for x,value in enumerate(vals):
                ax.annotate(f'{value:.4f}',(x,value),xytext=(0,9 if propagation else -15),textcoords='offset points',ha='center',fontsize=9)
        ax.set(xticks=[0,1],xticklabels=['Fusion off','Fusion on'],ylabel=INPUT['factorial_metric'],title=ds)
        ax.set_xlim(-.25,1.25);ax.margins(y=.45);ax.legend(fontsize=8,loc='upper left')
    save(fig,'05_affinity_factorial','Factorial comparison of affinity use in propagation and fusion on the same backbone. The full configuration (11) has the highest reported Recall@10, while the interaction contrast y11 − y10 − y01 + y00 is +0.0001, −0.0002, and +0.0001 on Baby, Sports, and Clothing, respectively. These are contrasts of the displayed point estimates, without seed-level uncertainty.')

    fig,axes=plt.subplots(1,3,figsize=(11.8,3.6),sharey=True)
    keys=['propagation_only','fusion_only','joint_gain','interaction']
    for ax,ds in zip(axes,DATASETS):
        vals=effects.set_index('dataset').loc[ds,keys].to_numpy(dtype=float)*1000
        ax.bar(range(4),vals,color=['#7197a8','#91aaa1','#c36a42','#93839c'],width=.62)
        for x,value in enumerate(vals): ax.text(x,value+(.045 if value>=0 else -.045),f'{value:+.1f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=10)
        ax.axhline(0,color='#52626a',lw=.8)
        ax.set(xticks=range(4),xticklabels=['P only','F only','Joint','Interaction'],title=ds,ylim=(-.4,1.9))
        ax.tick_params(axis='x',labelsize=8);ax.grid(axis='x',visible=False)
    axes[0].set_ylabel('Recall@10 difference (× 10⁻³)')
    save(fig,'06_affinity_effects','Propagation-only (y10 − y00), fusion-only (y01 − y00), joint (y11 − y00), and interaction (y11 − y10 − y01 + y00) contrasts. All differences are in Recall@10 units scaled by 10³. Positive joint gains coexist with a negative interaction on Sports. Bars do not represent uncertainty or significance.')


def source_and_changes(perf,ab):
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(SOURCE) as z: tree=etree.fromstring(z.read('word/document.xml'))
    body=tree.find('w:body',ns)
    blocks={i:''.join(el.xpath('.//w:t/text()',namespaces=ns)) for i,el in enumerate(body)}
    diffs=[]
    def cells(row): return [''.join(c.xpath('.//w:t/text()',namespaces=ns)) for c in row.findall('w:tc',ns)]
    candidates=perf.set_index(['model','dataset'])
    oldmodels=[]
    for row in body[148].findall('w:tr',ns)[2:-1]:
        c=cells(row);model=c[0].split(' (')[0];oldmodels.append(model)
        for di,ds in enumerate(DATASETS):
            for mi,metric in enumerate(METRICS):
                old=float(c[1+4*di+mi]);new=float(candidates.loc[(model,ds),metric])
                if old!=new: diffs.append(dict(table='Table 2',model=model,dataset=ds,metric=metric,old=old,new=new))
    variant=None
    for row in body[160].findall('w:tr',ns)[1:]:
        c=cells(row)
        if c[0]: variant='Full' if c[0]=='AMORE (Full)' else c[0]
        metric=c[1]
        for di,ds in enumerate(DATASETS):
            old=float(c[di+2]);new=float(ab[(ab.dataset==ds)&(ab.variant==variant)].iloc[0][metric])
            if old!=new: diffs.append(dict(table='Table 3',model=variant,dataset=ds,metric=metric,old=old,new=new))
    export(pd.DataFrame(diffs),'manuscript_numeric_changes')
    return blocks,diffs


def table_html(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def figure_html(name):
    f=next(f for f in FIGURES if f['name']==name)
    data=base64.b64encode((ROOT/'figures'/(name+'.png')).read_bytes()).decode()
    return '<figure><img src="data:image/png;base64,'+data+'" alt="'+html.escape(name)+'"><figcaption>'+html.escape(f['caption'])+'</figcaption><p class="downloads">'+ ' · '.join('<a href="figures/'+name+'.'+ext+'" download>'+ext.upper()+'</a>' for ext in ['png','svg','pdf'])+'</p></figure>'


def latex(perf,ab,effects,gains):
    lines=[r'% Requires booktabs, graphicx. Values are the supplied point estimates.',r'\begin{table*}[t]',r'\centering',r'\caption{Ranking performance on three Amazon datasets. Bold and underlined values indicate the best and second-best reported scores. Relative changes use the strongest baseline for each column.}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{l'+'r'*12+'}',r'\toprule',r'Model & \multicolumn{4}{c}{Baby} & \multicolumn{4}{c}{Sports} & \multicolumn{4}{c}{Clothing} \\',r' & '+' & '.join(['R@10','R@20','N@10','N@20']*3)+r' \\',r'\midrule']
    for label in INPUT['performance']:
        model=label.split(' (')[0];values=[]
        for ds in DATASETS:
            sub=perf[perf.dataset==ds].set_index('model')
            for metric in METRICS:
                v=sub.loc[model,metric];ranked=sorted(sub[metric].unique(),reverse=True)
                text=f'{v:.4f}'
                if v==ranked[0]: text=r'\textbf{'+text+'}'
                elif v==ranked[1]: text=r'\underline{'+text+'}'
                values.append(text)
        lines.append(label+' & '+' & '.join(values)+r' \\')
    lines.extend([r'\midrule',r'Improv. (\%) & '+' & '.join(f'{r.relative_percent:+.2f}' for r in gains.itertuples())+r' \\',r'\bottomrule',r'\end{tabular}}',r'\end{table*}'])
    (ROOT/'tables/table2_performance.tex').write_text('\n'.join(lines)+'\n')
    for name,headers,rows,caption in [
        ('table3_ablation',['Variant','Metric']+DATASETS,
         [[v,m]+[f'{ab[(ab.variant==v)&(ab.dataset==ds)].iloc[0][m]:.4f}' for ds in DATASETS] for v in INPUT['ablation'] for m in INPUT['ablation_metrics']],
         'Module ablations on three Amazon datasets.'),
        ('table4_affinity',['Dataset','00','10','01','11',r'$\Delta_{\mathrm{int}}$'],
         [[r.dataset]+[f'{getattr(r,k):.4f}' for k in ['y00','y10','y01','y11']]+[f'{r.interaction:+.4f}'] for r in effects.itertuples()],
         'Affinity-use factorial comparison in Recall@10. Interaction contrasts are calculated from the displayed values.')]:
        tex=[r'\begin{table}[t]',r'\centering',r'\caption{'+caption+'}',r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
        tex+=[' & '.join(row)+r' \\' for row in rows]
        tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
        (ROOT/'tables'/(name+'.tex')).write_text('\n'.join(tex)+'\n')


def validate(perf,ab,factor,gains,drops,effects,blocks):
    assert len(perf)==39 and len(ab)==12 and len(factor)==12
    assert perf[METRICS].gt(0).all().all() and perf[METRICS].lt(1).all().all()
    assert (perf['Recall@20']>=perf['Recall@10']).all()
    assert (perf['NDCG@20']>=perf['NDCG@10']).all()
    np.testing.assert_allclose(gains.relative_percent.round(2),INPUT['supplied_improvement_percent'],atol=1e-10)
    assert (gains.absolute_difference>0).sum()==11
    for ds in DATASETS:
        full=perf[(perf.dataset==ds)&(perf.model=='AMORE')].iloc[0]
        a=ab[(ab.dataset==ds)&(ab.variant=='Full')].iloc[0]
        np.testing.assert_allclose(full[INPUT['ablation_metrics']].astype(float),a[INPUT['ablation_metrics']].astype(float))
        np.testing.assert_allclose(full[INPUT['factorial_metric']],factor[(factor.dataset==ds)&(factor.variant=='11')].iloc[0].value)
    np.testing.assert_allclose(effects.interaction,[.0001,-.0002,.0001],atol=1e-12)
    assert (drops.absolute_drop>0).all()
    changes=json.loads((ROOT/'sentence_replacements.json').read_text())
    for change in changes:
        assert change['original'] in blocks[change['body_block']],change['id']
    return dict(status='passed',performance_rows=39,ablation_rows=12,factorial_rows=12,
                improvement_checks=12,highest_amore_scores=11,source_quote_checks=len(changes),
                uncertainty='Not calculated: seed-level records and repetition counts unavailable.',
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())


def main():
    for directory in ['data','figures','tables']: (ROOT/directory).mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,
                         'axes.axisbelow':True,'svg.fonttype':'none','pdf.fonttype':42})
    perf,ab,factor,gains,drops,effects=tables()
    plots(perf,ab,factor,gains,drops,effects)
    blocks,diffs=source_and_changes(perf,ab)
    from manuscript_text import write_guidance, write_report
    changes=write_guidance(ROOT,blocks,diffs,gains,drops,effects,FIGURES)
    latex(perf,ab,effects,gains)
    write_report(ROOT,INPUT,perf,ab,gains,drops,effects,FIGURES,changes,table_html,figure_html)
    audit=validate(perf,ab,factor,gains,drops,effects,blocks)
    (ROOT/'validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'manifest.json').write_text(json.dumps(dict(source=INPUT['source'],figures=FIGURES,
        inputs_sha256=hashlib.sha256((ROOT/'inputs.json').read_bytes()).hexdigest(),
        dependencies={'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
