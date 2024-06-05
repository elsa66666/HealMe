import numpy as np
import matplotlib.pyplot as plt

# categories = ['Interested', 'Excited', 'Strong', 'Enthusiastic', 'Proud',
# 'Alert', 'Inspired', 'Determined', 'Attentive', 'Active']
categories = ['Distressed', 'Upset', 'Guilty', 'Scared', 'Hostile',
              'Irritable', 'Ashamed', 'Nervous', 'Jittery', 'Afraid']
categories = [i for i in range(1, 11)]
# scores_before = [5, 5, 1, 1, 5, 5, 1, 2, 4, 4]
# scores_after = [1, 1, 1, 1, 3, 1, 1, 1, 1, 1]
scores_before = [0 for _ in range(10)]
scores_after = [0 for _ in range(10)]

# Number of variables we're plotting.
num_vars = len(categories)

# Split the circle into even parts and save the angles
# so we know where to put each axis.
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is made circular, so we need to "complete the loop"
# and append the start to the end.
scores_before += scores_before[:1]
scores_after += scores_after[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw the outline of our data.
ax.plot(angles, scores_before, color='red', linewidth=2, label='Before')
ax.fill(angles, scores_before, color='red', alpha=0.25)

ax.plot(angles, scores_after, color='blue', linewidth=2, label='After')
ax.fill(angles, scores_after, color='blue', alpha=0.25)

# Fix axis to go in the right order and start at 12 o'clock.
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label.
ax.set_thetagrids(np.degrees(angles[:-1]), categories)

# Go through labels and adjust alignment based on where
# it is in the circle.
for label, angle in zip(ax.get_xticklabels(), angles):
    if angle in (0, np.pi):
        label.set_horizontalalignment('center')
    elif 0 < angle < np.pi:
        label.set_horizontalalignment('right')
    else:
        label.set_horizontalalignment('left')

# Ensure radar goes from 0 to 5.
ax.set_ylim(0, 5)

# Add some custom styling.
# Change the color of the tick labels.
ax.tick_params(colors='#222222', labelsize=25)
# Make the y-axis (0-5) labels smaller.
ax.tick_params(axis='y', labelsize=8)
# Change the color of the circular gridlines.
ax.grid(color='#AAAAAA')
# Change the color of the outermost gridline (the spine).
ax.spines['polar'].set_color('#222222')
# Change the background color inside the circle itself.
ax.set_facecolor('#FAFAFA')

# Add a legend as well.
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()
fig_path = 'circle.png'
fig.savefig(fig_path, bbox_inches='tight')
