"""Reproduce the article figures: python3 scripts/plot_norms.py."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle, Rectangle

OUT = Path(__file__).resolve().parents[1] / 'public' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 15,
                     'axes.labelsize': 11, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
BLUE, TEAL, GOLD, GRAY = '#3852c4', '#138579', '#b17624', '#83909e'
COLORS = [BLUE, TEAL, GOLD]

def plane(ax, limit=1.35, xlabel='$x_1$', ylabel='$x_2$'):
    ax.set_aspect('equal'); ax.set_xlim(-limit,limit); ax.set_ylim(-limit,limit)
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1])
    ax.axhline(0,color='#d8dee7',lw=.9,zorder=0);ax.axvline(0,color='#d8dee7',lw=.9,zorder=0)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(length=0,labelsize=10,pad=6,colors='#667080')
    ax.set_xlabel(xlabel,labelpad=7); ax.set_ylabel(ylabel,rotation=0,labelpad=14)

def ball(ax,kind,color):
    if kind==1: shape=Polygon([[1,0],[0,1],[-1,0],[0,-1]],closed=True)
    elif kind==2: shape=Circle((0,0),1)
    else: shape=Rectangle((-1,-1),2,2)
    shape.set(facecolor=color+'13',edgecolor=color,linewidth=2.2,zorder=2)
    ax.add_patch(shape)

def save(fig,name):
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',pad_inches=.14)
    plt.close(fig)

for mobile in [False,True]:
    fig,axes=plt.subplots(3 if mobile else 1,1 if mobile else 3,figsize=(5,12) if mobile else (12,3.9),layout='constrained')
    for ax,p,c,description in zip(axes,[1,2,np.inf],COLORS,['shared absolute-value budget','shared squared-length budget','one bound per coordinate']):
        plane(ax);ball(ax,p,c)
        label='\\infty' if np.isinf(p) else str(p)
        ax.set_title(rf'$\|x\|_{{{label}}}\leq 1$',color=c,pad=16)
        ax.text(.5,-.25,description,transform=ax.transAxes,ha='center',fontsize=10,color='#606b7a')
    save(fig,'vector-balls-mobile' if mobile else 'vector-balls')

A=np.array([[1.,2.],[0.,2.]])
t=np.linspace(0,2*np.pi,700)
x=np.vstack([np.cos(t),np.sin(t)])
y=A@x
U,s,Vt=np.linalg.svd(A)
for mobile in [False,True]:
    fig,axes=plt.subplots(2 if mobile else 1,1 if mobile else 2,figsize=(5.5,10) if mobile else (11,4.6),layout='constrained')
    ax=axes[0];plane(ax);ax.plot(*x,color=TEAL,lw=2.2)
    v=Vt[0];ax.arrow(0,0,*v,width=.013,head_width=.09,length_includes_head=True,color=BLUE,zorder=4)
    ax.text(v[0]-.42,v[1]+.13,r'$v_1$',color=BLUE)
    ax.set_title('Unit inputs',pad=17)
    ax=axes[1];plane(ax,3.4,xlabel='$y_1$',ylabel='$y_2$')
    ax.set_xticks([-3,-1,0,1,3]);ax.set_yticks([-3,-1,0,1,3])
    ax.plot(*y,color=TEAL,lw=2.2)
    ax.plot(*x,color=GRAY,lw=1.1,ls='--',alpha=.6)
    for j,c in [(1,GRAY),(0,BLUE)]:
        point=U[:,j]*s[j]
        ax.arrow(0,0,*point,width=.025,head_width=.15,length_includes_head=True,color=c,zorder=4)
    ax.set_title(r'$y=Ax$: an ellipse',pad=17)
    ax.text(.04,.95,rf'$\sigma_1={s[0]:.3f},\quad\sigma_2={s[1]:.3f}$',transform=ax.transAxes,va='top',fontsize=12,color=BLUE)
    save(fig,'matrix-map-mobile' if mobile else 'matrix-map')

for mobile in [False,True]:
    fig,axes=plt.subplots(2 if mobile else 1,1 if mobile else 2,figsize=(5,8.7) if mobile else (10,4.2),layout='constrained')
    for ax,p,c,title in zip(axes,[np.inf,2],[BLUE,TEAL],[r'Spectral: $\max(|a|,|b|)\leq1$',r'Frobenius: $a^2+b^2\leq1$']):
        plane(ax,1.4,xlabel='$a$',ylabel='$b$');ball(ax,p,c);ax.set_title(title,fontsize=13,pad=17)
        ax.scatter([1],[1],color=GOLD,s=35,zorder=5)
        ax.annotate('$I=\\mathrm{diag}(1,1)$',(1,1),xytext=(.1,1.21),fontsize=10,color=GOLD)
    save(fig,'matrix-balls-mobile' if mobile else 'matrix-balls')

G=np.array([3.,4.]); steps=[np.array([0.,-1.]),-G/np.linalg.norm(G),-np.ones(2)]
for mobile in [False,True]:
    fig,axes=plt.subplots(3 if mobile else 1,1 if mobile else 3,figsize=(5,12.5) if mobile else (12,4.1),layout='constrained')
    for ax,p,c,d in zip(axes,[1,2,np.inf],COLORS,steps):
        plane(ax,1.45,xlabel='$\\Delta_1$',ylabel='$\\Delta_2$');ball(ax,p,c)
        xx=np.linspace(-1.45,1.45,150)
        for level in [-7,-5,-4,-2,0,2,4]:
            ax.plot(xx,(level-3*xx)/4,color=GRAY,lw=.7,alpha=.36,zorder=1)
        value=G@d
        ax.plot(xx,(value-3*xx)/4,color=c,lw=1.4,ls='--',zorder=3)
        ax.arrow(0,0,*d,width=.018,head_width=.1,length_includes_head=True,color=c,zorder=5)
        ax.scatter(*d,color=c,s=38,zorder=6)
        label='\\infty' if np.isinf(p) else str(p)
        ax.set_title(rf'$\|\Delta\|_{{{label}}}\leq1$',color=c,pad=17)
        ax.text(.5,-.23,rf'$\Delta=({d[0]:g}, {d[1]:g}),\quad g^\top\Delta={value:g}$',ha='center',transform=ax.transAxes,fontsize=11,color=c)
    save(fig,'steepest-steps-mobile' if mobile else 'steepest-steps')
print('Generated eight SVGs (four figures with mobile variants).')
