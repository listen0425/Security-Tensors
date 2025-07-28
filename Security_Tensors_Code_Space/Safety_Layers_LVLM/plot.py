import matplotlib.pyplot as plt
import pickle
import numpy as np
import fire

def calcu(data_path,cross_attn):
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)
    good=np.array(datas[0])
    good = [good[i] for i in range(len(good)) if i not in cross_attn]
    mean_good= np.mean(good, axis=0)
    mean_good[mean_good>1]=1
    mean_good=np.array([mean_good[i] for i in range(len(mean_good)) if i not in cross_attn])
    std_good = np.std(good, axis=0)
    std_good=np.array([std_good[i] for i in range(len(std_good)) if i not in cross_attn])

    bad=np.array(datas[2])
    bad = [bad[i] for i in range(len(bad)) if i not in cross_attn]
    mean_bad= np.mean(bad, axis=0)
    mean_bad[mean_bad>1]=1
    mean_bad=np.array([mean_bad[i] for i in range(len(mean_bad)) if i not in cross_attn])
    std_bad = np.std(bad, axis=0)
    std_bad=np.array([std_bad[i] for i in range(len(std_bad)) if i not in cross_attn])

    angle_a = np.degrees(np.arccos(mean_good))
    angle_b = np.degrees(np.arccos(mean_bad))
    angle_diff = np.abs(angle_a - angle_b)
    print(mean_good)
    print(mean_bad)
    print(angle_diff)
    return mean_good,mean_bad,angle_diff


def main(
    data_path1: str='pkls/textual_tensor_activation.pkl',
    data_path2: str='pkls/visual_tensor_activation.pkl',
    save_dir: str='Activation.png',
):
    cross_attn=[3,8,13,18,23,28,33,38]
    cross_attn=[]
    x=range(40)
    mean_good,mean_bad,angle_diff=calcu(data_path1,cross_attn=cross_attn)
    
    mean_good2,mean_bad2,angle_diff2=calcu(data_path2,cross_attn=cross_attn)
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 3.5), sharex='col', sharey='row')

    for ax in axes.flat:
        ax.set_facecolor('#f5f5f5')  

    plt.subplots_adjust(wspace=0.15, hspace=0.25)


    label_fontsize = 8
    tick_fontsize = 8
    legend_fontsize = 8
    axes[0, 1].plot(x, mean_good, label='N-N+$\delta_{v}$ pairs', color='tab:blue', linewidth=1.5)
    axes[0, 1].plot(x, mean_bad, label='M-M+$\delta_{v}$ pairs', color='#2ca02c', linewidth=1.5)
    axes[0, 0].set_ylabel('Cos_sim Value', fontsize=label_fontsize)
    axes[0, 1].tick_params(labelsize=tick_fontsize)
    axes[0, 1].legend(fontsize=legend_fontsize, loc='lower left', frameon=False)
    axes[0, 1].grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    axes[1, 1].plot(x, angle_diff, color='#d62728', label="Angular Difference", linewidth=1.5)
    axes[1, 0].set_ylabel('Angle Degree Value', fontsize=label_fontsize)
    axes[1, 1].tick_params(labelsize=tick_fontsize)
    axes[1, 1].legend(fontsize=legend_fontsize, loc='upper left', frameon=False)
    axes[1, 1].grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    axes[0, 0].plot(x, mean_good2, label='N-N+$\delta_{t}$ pairs', color='tab:blue', linewidth=1.5)
    axes[0, 0].plot(x, mean_bad2, label='M-M+$\delta_{t}$ pairs', color='#2ca02c', linewidth=1.5)
    axes[0, 0].tick_params(labelsize=tick_fontsize)
    axes[0, 0].legend(fontsize=legend_fontsize, loc='lower left', frameon=False)
    axes[0, 0].grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    axes[1, 0].plot(x, angle_diff2, label="Angular Difference", color='#d62728', linewidth=1.5)
    axes[1, 0].tick_params(labelsize=tick_fontsize)
    axes[1, 0].legend(fontsize=legend_fontsize, loc='upper left', frameon=False)
    axes[1, 0].grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    axes[0, 0].set_title(" ", fontsize=9)
    # axes[0, 1].set_title("LLaMA-3.2-Vision", fontsize=9)


    fig.text(0.53, 0.92, 'Layer-wise Security Tensors Activation Analysis', ha='center', fontsize=9)
    
    

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)   
        ax.spines['bottom'].set_visible(False) 
        ax.tick_params(left=False, bottom=False)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_dir, bbox_inches='tight', dpi=500)
fire.Fire(main)