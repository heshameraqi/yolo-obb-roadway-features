import matplotlib.pyplot as plt
import math
import numpy

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
color_map = plt.get_cmap('bwr')
length = 11
width = 8
samples_angle_step = 15
outline_width = 2

# ------------------------------------------------------------------------------
# Member Functions
# ------------------------------------------------------------------------------
def get_color(value, min, max):
    # min_val = -1.0
    # max_val = 1.0
    return color_map((value-min)/(max-min))  # color_map((value-min)/(max-min+0.0001))

def drawline(ax, endx, endy, text_x, text_y, color):
    [endx_1, endx_2] = endx
    [endy_1, endy_2] = endy
    ax.plot([endx_1, endx_2], [endy_1, endy_2], color='black', linewidth=width+outline_width)  # Outline
    ax.plot([endx_1, endx_2], [endy_1, endy_2], color=color, linewidth=width)
    ax.text(text_x, text_y, str(angles[i]) + "°", horizontalalignment='center', verticalalignment='center', fontsize=6)

# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
x, y = (0, 0)
fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
axs = axs.ravel()
for ax in axs:
    ax.set_ylim([-13, 13])  # set the bounds to be 10, 10
    ax.set_xlim([-13, 13])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

angles = range(0, 180, samples_angle_step)
for exp in range(5):  #range(7)
    flag = True  # To indicate angle text spacing from lines
    for i in range(len(angles)):
        endy_1 = length * math.sin(math.radians(angles[i]))
        endx_1 = length * math.cos(math.radians(angles[i]))
        endy_2 = -length * math.sin(math.radians(angles[i]))
        endx_2 = -length * math.cos(math.radians(angles[i]))
        text_x = (length + .5 * flag + 1.4) * math.cos(math.radians(angles[i]))
        text_y = (length + .5 * flag + 1.4) * math.sin(math.radians(angles[i]))
        # flag = not flag  # One near, one far, and so on ...
        if exp == 0:  # Sin Theta
            # axs[exp].title.set_text('Θ\nDiscontinuity problem around 0°')
            min_val = 0
            max_val = 1
            drawline(axs[exp], [endx_1, endx_2], [endy_1, endy_2], text_x, text_y,
                     color=get_color(angles[i]/180., min_val, max_val))
            axs[exp].title.set_text(' Θ encoding')
        elif exp == 1:
            # axs[exp].title.set_text('sin(Θ)\nSymmetric around 90°,\nrequires additional variable')
            axs[exp].title.set_text('sin(Θ) encoding')
            min_val = min([math.sin(math.radians(ang)) for ang in angles])
            max_val = max([math.sin(math.radians(ang)) for ang in angles])
            drawline(axs[exp], [endx_1, endx_2], [endy_1, endy_2], text_x, text_y,
                     color=get_color(math.sin(math.radians(angles[i])), min_val, max_val))
        elif exp == 2:
            # axs[exp].title.set_text('cos(Θ)\nDiscontinuity problem around 0°')
            axs[exp].title.set_text('cos(Θ) encoding')
            min_val = min([math.cos(math.radians(ang)) for ang in angles])
            max_val = max([math.cos(math.radians(ang)) for ang in angles])
            drawline(axs[exp], [endx_1, endx_2], [endy_1, endy_2], text_x, text_y,
                     color=get_color(math.cos(math.radians(angles[i])), min_val, max_val))
        elif exp == 3:
            # axs[exp].title.set_text('sin(2Θ)\nSymmetric around 45° and 135°,\nrequires additional variable')
            axs[exp].title.set_text('sin(2Θ) encoding')
            min_val = min([math.sin(math.radians(2*ang)) for ang in angles])
            max_val = max([math.sin(math.radians(2*ang)) for ang in angles])
            drawline(axs[exp], [endx_1, endx_2], [endy_1, endy_2], text_x, text_y,
                     color=get_color(math.sin(math.radians(2*angles[i])), min_val, max_val))
        elif exp == 4:
            # axs[exp].title.set_text('cos(2Θ)\nSymmetric around 90°,\nrequires additional variable')
            axs[exp].title.set_text('cos(2Θ) encoding')
            min_val = min([math.cos(math.radians(2*ang)) for ang in angles])
            max_val = max([math.cos(math.radians(2*ang)) for ang in angles])
            drawline(axs[exp], [endx_1, endx_2], [endy_1, endy_2], text_x, text_y,
                     color=get_color(math.cos(math.radians(2*angles[i])), min_val, max_val))
    # Draw colorbar
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm._A = []  # fake up the array of the scalar mappable
    axs[exp].get_figure().colorbar(sm, ax=axs[exp], orientation='horizontal', fraction=0.025, pad=-0.01)  # , pad=0.2
    axs[exp].set_aspect('equal')
plt.tight_layout()

# Draw subfigrue separator lines
width = 3
offset = 0.03
import numpy as np
import matplotlib.transforms as mtrans
r = fig.canvas.get_renderer()
get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
bboxes = np.array(list(map(get_bbox, axs.flat)), mtrans.Bbox).reshape(axs.shape)
xmax = list(map(lambda b: b.x1, bboxes.flat))
xmin = list(map(lambda b: b.x0, bboxes.flat))
ymax = list(map(lambda b: b.y1, bboxes.flat))
ymin = list(map(lambda b: b.y0, bboxes.flat))

# Vertical lines
x_centre = [xmax[0] + (xmin[1]-xmax[0])/2.0, xmax[2]+(xmin[3]-xmax[2])/2.0]
for x in x_centre:
    line = plt.Line2D([x, x], [ymin[0]-2.5*offset, ymax[-1]+offset], transform=fig.transFigure, color="black", linewidth=width)
    fig.add_artist(line)
line = plt.Line2D([xmin[0]-0.3*offset, xmin[0]-0.3*offset], [ymin[0]-2.5*offset, ymax[-1]+offset], transform=fig.transFigure, color="black", linewidth=width)
fig.add_artist(line)
line = plt.Line2D([xmax[-1]+0.3*offset, xmax[-1]+0.3*offset], [ymin[0]-2.5*offset, ymax[-1]+offset], transform=fig.transFigure, color="black", linewidth=width)
fig.add_artist(line)

# Horizontal Lines
line = plt.Line2D([xmin[0]-0.3*offset, xmax[-1]+0.3*offset], [ymin[0]-2.5*offset, ymin[0]-2.5*offset], transform=fig.transFigure, color="black", linewidth=width)
fig.add_artist(line)
line = plt.Line2D([xmin[0]-0.3*offset, xmax[-1]+0.3*offset], [ymax[-1]+offset, ymax[-1]+offset], transform=fig.transFigure, color="black", linewidth=width)
fig.add_artist(line)

# Save and show image
plt.savefig('theta_encoding.eps', bbox_inches="tight", format='eps')
plt.show(block=False)


