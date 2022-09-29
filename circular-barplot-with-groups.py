import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
rng = np.random.default_rng(123)

# Build a dataset
df = pd.read_csv('/home/xumengdie/tmp/dualAtt/colddrugcomb.csv')
# Show 3 first rows
print(df.head(3))

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
    padding = 0.04
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor"
        ) 



VALUES = df["value"].values
LABELS = df["name"].values
GROUP = df["group"].values

PAD = 3
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)
OFFSET = np.pi / 2

offset = 0
IDXS = []
GROUPS_SIZE = [7]*np.unique(GROUP).shape[0]
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + int(size) + PAD))#range加0，
    offset += size + PAD

fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": "polar"})
ax.set_theta_offset(OFFSET)
ax.set_ylim(0, 1)
ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])


# COLORS = [f"C{i}" for i, size in enumerate(GROUPS_SIZE) for _ in range(size)]
COLORS = ["#6C5B7B","#C06C84","#9999cc","#339999","#996600","#666600","#F8B195"] * np.unique(GROUP).shape[0]
 
line = ax.bar(
    ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS,
    edgecolor="white", linewidth=2
)
print(IDXS, VALUES, LABELS)

# add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

# Extra customization below here --------------------

# This iterates over the sizes of the groups adding reference
# lines and annotations.

offset = 0 
# for group, size in zip(['ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC', 'Kappa', 'AP'], GROUPS_SIZE):
for group, size in zip(['ACC', 'BACC', 'Prec', 'Rec', 'F1', 'MCC', 'AUC', 'Kappa', 'AP'], GROUPS_SIZE):
    # Add line below bars
    x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
    ax.plot(x1, [0.1] * 50, color="#333333")
    
    # Add text to indicate group
    ax.text(
        np.mean(x1), 0.98, group, color="#333333", fontsize=14, 
        fontweight="bold", ha="center", va="center"
    )
    
    # Add reference lines at 20, 40, 60, and 80
    x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
    ax.plot(x2, [0.2] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.4] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.6] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.8] * 50, color="#bebebe", lw=0.8)
    # ax.plot(x2, [0.3] * 50, color="#bebebe", lw=0.8)
    # ax.plot(x2, [0.6] * 50, color="#bebebe", lw=0.8)
    # ax.plot(x2, [0.9] * 50, color="#bebebe", lw=0.8)
    #
    offset += size + PAD

ax.text(
            x=ANGLES[2],
            y=0.2,
            s='0.2',
            va="center",
            rotation_mode="anchor"
        )
ax.text(
            x=ANGLES[2],
            y=0.4,
            s='0.4',
            va="center",
            rotation_mode="anchor"
        )
ax.text(
            x=ANGLES[2],
            y=0.6,
            s='0.6',
            va="center",
            rotation_mode="anchor"
        )
ax.text(
            x=ANGLES[2],
            y=0.8,
            s='0.8',
            va="center",
            rotation_mode="anchor"
        )
# ax.text(
#             x=ANGLES[2],
#             y=0.3,
#             s='0.3',
#             va="center",
#             rotation_mode="anchor"
#         )
# ax.text(
#             x=ANGLES[2],
#             y=0.6,
#             s='0.6',
#             va="center",
#             rotation_mode="anchor"
#         )
# ax.text(
#             x=ANGLES[2],
#             y=0.9,
#             s='0.9',
#             va="center",
#             rotation_mode="anchor"
#         )
# ["#6C5B7B","#C06C84","#F67280","#F8B195"]
ulabels = np.unique(LABELS)
colors = { 'DeepDDS':'#6C5B7B',   'MRGNN':'#C06C84', 'DeepSynergy':'#9999cc','MatchMaker':'#339999','GCNBMP':'#996600','EPGCNDS':'#666600','DFFNDDS':'#F8B195' }
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels, fontsize=10)

plt.savefig('/home/xumengdie/tmp/dualAtt/circular-plotcomb_COLD5.png',bbox_inches='tight')
print(np.unique(GROUP))