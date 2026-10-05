# import nibabel as nib
# import numpy as np
#
# ct_img = nib.load("xxx/path")
# heart_img = nib.load("xxx/path")
#
# ct = ct_img.get_fdata()
# heart = heart_img.get_fdata().astype(int)
#
# if ct.shape != heart.shape:
#     raise ValueError("CT 和 mask 大小不一致")
#
# ct_new = ct.copy()
# max_val = np.max(ct)
#
# # 心脏区域提亮
# ct_new[heart > 0] = max_val
#
# out_path = "xxx/path"
#
# new_img = nib.Nifti1Image(ct_new.astype(np.float32), affine=ct_img.affine, header=ct_img.header)
# nib.save(new_img, out_path)
# print("高亮心脏的 CT 已保存：", out_path)
#


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 数据 ----------
data = [
    ("11", 0.783, 0.168), ("12", 0.789, 0.180), ("13", 0.841, 0.170),
    ("14", 0.774, 0.146), ("15", 0.837, 0.164), ("16", 0.744, 0.183), ("17", 0.908, 0.240),
    ("21", 0.774, 0.181), ("22", 0.770, 0.154), ("23", 0.810, 0.156),
    ("24", 0.762, 0.152), ("25", 0.841, 0.183), ("26", 0.720, 0.190), ("27", 0.842, 0.184),
    ("31", 0.793, 0.170), ("32", 0.816, 0.162), ("33", 0.805, 0.174),
    ("34", 0.774, 0.165), ("35", 0.806, 0.170), ("36", 0.747, 0.159), ("37", 0.861, 0.190),
    ("41", 0.773, 0.157), ("42", 0.804, 0.151), ("43", 0.776, 0.178),
    ("44", 0.802, 0.186), ("45", 0.793, 0.184), ("46", 0.753, 0.164), ("47", 0.856, 0.218),
]
df = pd.DataFrame(data, columns=["FDI_tooth", "IoU_mean", "Delta_up"])

# ---------- 造 violin 样本 ----------
rng = np.random.default_rng(102)

def make_violin_samples(means, sigma, n, clip=None):
    samples = []
    for m in means:
        s = rng.normal(loc=m, scale=sigma, size=n)
        if clip is not None:
            s = np.clip(s, *clip)
        samples.append(s)
    return samples

iou_samples = make_violin_samples(df["IoU_mean"].to_numpy(), 0.03, 300, clip=(0, 0.99))

# ---------- 画图 ----------
fig, ax = plt.subplots(figsize=(16, 4.5))

# 1) 画 violin（默认纵向）
parts = ax.violinplot(iou_samples,
                      positions=np.arange(1, len(df)+1),
                      showmeans=True, showmedians=False, widths=0.7)

# 2) 给每个 violin 上色
color_map = {str(i): f"C{i-1}" for i in range(1, 8)}   # 1->C0, 2->C1 …
for i, pc in enumerate(parts['bodies']):
    tail_num = df["FDI_tooth"].iloc[i][-1]
    pc.set_facecolor(color_map[tail_num])
    pc.set_edgecolor('black')

# 3) 手动加四分位线（纵向）
for i, data in enumerate(iou_samples, 1):
    q1, med, q3 = np.percentile(data, [25, 50, 75])

    gap = 0.1          # 框的横向“半宽”，自己看着调
    lw_box = 1.0        # 框线粗细
    color_box = 'black'

    # 左竖线
    ax.plot([i-gap, i-gap], [q1, q3], color=color_box, lw=lw_box)
    # 右竖线
    ax.plot([i+gap, i+gap], [q1, q3], color=color_box, lw=lw_box)
    # 上横线
    ax.plot([i-gap, i+gap], [q3, q3], color=color_box, lw=lw_box)
    # 下横线
    ax.plot([i-gap, i+gap], [q1, q1], color=color_box, lw=lw_box)

    # 中位数短横线（白色）
    # ax.plot([i-gap, i+gap], [med, med], color='white', lw=1.5)
    # 中位横线（稍微出头好看）
    # ax.plot([i-0.1, i+0.1], [med, med], color='white', lw=2)


# 设置标题和标签
ax.set_ylabel("3D-IoU")
ax.set_xticks(np.arange(1, len(df) + 1))
ax.set_xticklabels(df["FDI_tooth"].tolist())

# 限制 x 轴的范围，缩小左右空白区域
ax.set_xlim(0.5, len(df) + 0.5)

ax.set_ylim(0.6, 1.0)  # Starting y-axis at 50% mark
ax.grid(True, axis="y", alpha=0.3)

# 显示图表
fig.tight_layout()
plt.show()

# Saving the modified plot
path1_modified = "xxx/path"
fig.savefig(path1_modified, dpi=600)
plt.close(fig)

print(f"Plot saved as {path1_modified}")