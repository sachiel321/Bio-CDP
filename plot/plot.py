import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
# Swtich to Type 42 Fonts.(A.K.A True Type)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def draw_plt(val, val_name, label=None):
    """将数据绘制成曲线图，val是数据，val_name是变量名称"""
    #plt.figure()
    plt.plot([i.step for i in val], [j.value for j in val], label=label)
    """横坐标是step，迭代次数
    纵坐标是变量值"""
    plt.xlabel('step')
    plt.ylabel(val_name)

def non_zero_mean(np_arr, axis=0):
    mean_len = np_arr.shape[1]
    mean = np.zeros(mean_len)
    for i in range(mean_len):
        temp = np_arr[:,i]
        temp = temp[np.nonzero(temp)]
        if len(temp)==0:
            mean[i] = 0
        else:
            mean[i] = temp.mean()
    return mean

def non_zero_var(np_arr, axis=0):
    var_len = np_arr.shape[1]
    var = np.zeros(var_len)
    for i in range(var_len):
        temp = np_arr[:,i]
        temp = temp[np.nonzero(temp)]
        if len(temp)==0:
            var[i] = 0
        else:
            var[i] = temp.var()
    return var

def RL_plot(ax,step,data,color,label,j):
    avg = non_zero_mean(data,axis=0)
    print(avg.shape)
    var = non_zero_var(data,axis=0)
    ci = 1.96*np.sqrt(var/data.shape[0])

    ax.plot(step[:j]/1e4, avg[:j], color=color,label=label)
    
    r1 = list(map(lambda x: (x[0]-x[1]), zip(avg, ci)))
    r2 = list(map(lambda x: (x[0]+x[1]), zip(avg, ci)))
    ax.fill_between(step[:j]/1e4, r1[0:j], r2[0:j], color=color, alpha=0.2)

def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

def load_data(data_1, data_2):

    step_sac = np.array(data_1['Step'])
    step_biocdp = np.array(data_2['Step'])
    reward_sac = np.zeros([3, step_sac.shape[0]])
    reward_casac = np.zeros([3, step_sac.shape[0]])
    reward_biocdp = np.zeros([3, step_biocdp.shape[0]])

    reward_sac[0, :] = np.array(data_1['SAC-RoboPianist-debug-CMajorScaleTwoHands-v0-44-1699612702.7078493--sac-v1_lr5e-4 - train/return'])
    reward_sac[1, :] = np.array(data_1['SAC-RoboPianist-debug-CMajorScaleTwoHands-v0-43-1699612636.2796128--sac-v1_lr5e-4 - train/return'])
    reward_sac[2, :] = np.array(data_1['SAC-RoboPianist-debug-CMajorScaleTwoHands-v0-42-1699605677.9389431--sac-v1_lr5e-4 - train/return'])

    reward_casac[0, :] = np.array(data_1['SAC-RoboPianist-debug-CMajorScaleTwoHands-v0-44-1699667573.305641--casac-v1_lr5e-4 - train/return'])
    reward_casac[1, :] = np.array(data_1['SAC-RoboPianist-debug-CMajorScaleTwoHands-v0-43-1699667513.3532948--casac-v1_lr5e-4 - train/return'])
    reward_casac[2, :] = np.array(data_1['SAC-RoboPianist-debug-CMajorScaleTwoHands-v0-42-1699667346.0416853--casac-v1_lr5e-4 - train/return'])

    reward_biocdp[0, :] = np.array(data_2['SAC-RoboPianist-debug-CMajorScaleTwoHands-v0-44-1699603969.8992152--dipo-clipaction-CMajorScaleTwoHands - train/return'])
    reward_biocdp[1, :] = np.array(data_2['SAC-RoboPianist-debug-CMajorScaleTwoHands-v0-43-1699603917.91163--dipo-clipaction-CMajorScaleTwoHands - train/return'])
    reward_biocdp[2, :] = np.array(data_2['SAC-RoboPianist-debug-CMajorScaleTwoHands-v0-42-1699603784.8798096--dipo-clipaction-CMajorScaleTwoHands - train/return'])
    
    return step_sac, step_biocdp,reward_sac, reward_casac, reward_biocdp

df_1 = pd.read_csv('D:/study_in_CASIA/博二/Project11/作图/plot/CMajorScaleTwoHands-sac-casac.csv')
df_2 = pd.read_csv('D:/study_in_CASIA/博二/Project11/作图/plot/CMajorScaleTwoHands-dipo.csv')
step_sac, step_biocdp,reward_sac, reward_casac, reward_biocdp = load_data(df_1, df_2)
map_name = 'CMajorScaleTwoHands'

f, ax = plt.subplots(1,1,figsize=(12, 8))
color = 'r'
# RL_plot(ax,IterClipVtrace_step,IterClipVtrace_value,color,'GPPO-SAG',j_IterClipVtrace)
RL_plot(ax,step_sac,reward_sac,color,'SAC',100)

color = 'g'
# RL_plot(ax,CombineVtrace_step,CombineVtrace_value,color,'AlphaStar',j_CombineVtrace)
RL_plot(ax,step_sac,reward_casac,color,'CASAC',100)


color = 'b'
RL_plot(ax,step_biocdp,reward_biocdp,color,'Bio-CDP',100)

plt.tick_params(labelsize=23)
plt.xlabel('Step×10k',fontsize=23)
plt.ylabel('Reward',fontsize=23)
plt.ylim((0.3,1.0))
plt.legend(fontsize=23, loc='upper left')
plt.show()
f.savefig(f'{map_name}_reward.pdf')