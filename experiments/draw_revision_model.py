#!/usr/bin/env python3
"""Draw the audited original-AMORE computation; contains no empirical results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.set(xlim=(-.3, 12.3), ylim=(-.4, 5))
    ax.axis('off')

    def box(x, y, title, color='#edf2f7', width=2.15):
        ax.add_patch(FancyBboxPatch((x, y), width, 1., boxstyle='round,pad=0.06',
                                   facecolor=color, edgecolor='#334155', linewidth=1.2))
        ax.text(x+width/2, y+.5, title, ha='center', va='center')

    def arrow(start, end, label=None, pos=None, dashed=False):
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='#334155',
                                                           linewidth=1.3, linestyle='--' if dashed else '-'))
        if label:
            ax.text(*(pos or ((start[0]+end[0])/2, (start[1]+end[1])/2+.13)), label,
                    ha='center', va='bottom', fontsize=9, bbox=dict(facecolor='white', edgecolor='none', pad=1.2))

    box(0, 3.1, 'Trainable item features\nand ID embeddings')
    box(3, 3.1, 'Modality gates +\nfixed item graphs\n' + r'$H_v, H_t;\ X_v, X_t$')
    box(0, .8, 'Observed UI edges\n+ input ID vectors\nPer-user softmax')
    box(3, .8, 'Threshold / re-attention\nPool by item\n' + r'$q_i$ and $r_i$')
    box(6.2, .8, 'Switch G: initialize\nitems using ' + r'$r_i$' + '\nGraph propagation ' + r'$C$', '#e5eef6')
    box(6.2, 3.1, 'Switch F: weights\nconditioned on ' + r'$q_i$' + '\nCoarse + residuals ' + r'$S$', '#e7f1e8')
    box(9.65, 1.95, r'$O=C+S$' + '\nUser-item dot product\nFull-catalogue ranking')
    arrow((2.15, 3.6), (3, 3.6))
    arrow((2.15, 1.3), (3, 1.3))
    arrow((5.15, 3.6), (6.2, 3.6))
    arrow((5.15, 1.3), (6.2, 1.3), r'$r_i$', dashed=True)
    arrow((5.15, 1.7), (6.2, 3.2), r'$q_i$', (5.65, 2.6), dashed=True)
    arrow((7.27, 1.8), (7.27, 3.1), r'$C$: behavior gates', (8.25, 2.3))
    arrow((8.35, 3.6), (9.65, 2.8), r'$S$', (9.05, 3.45))
    arrow((8.35, 1.3), (9.65, 2.2), r'$C$', (9.05, 1.35))
    ax.text(3.1, 2.4, 'Modal mean M enters\nre-attention and initialization', ha='center', fontsize=9)
    ax.text(6, 4.65, 'Original AMORE: independently test propagation and fusion affinity', ha='center', fontsize=13)
    ax.text(6, .1, 'Training adds extended views, contrastive objectives, scalar moments and representation penalties.', ha='center', fontsize=10)
    ax.text(6, -.22, 'G/F = 00, 10, 01, 11. Dashed arrows denote the two controlled uses of pooled affinity.', ha='center', fontsize=9)
    output = ROOT / 'revision/figures'
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(output / ('model.' + suffix), dpi=300, bbox_inches='tight')
    svg = output / 'model.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
