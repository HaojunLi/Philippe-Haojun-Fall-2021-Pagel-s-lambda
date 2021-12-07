import matplotlib.pyplot as plt

# create a figure and axes handle
fig, ax = plt.subplots()
# Function to print mouse click event coordinates
def onclick(event):
   print([event.xdata, event.ydata])

# Bind the button_press_event with the onclick() method
fig.canvas.mpl_connect('button_press_event', onclick)
# plot data on the main axes
ax.plot([1,2,3],[9,5,4])
ax.set_ylim(0,10)
ax.set_xlim(-1, 8)
# create an inset axes
# the input array specifies the sizing and position of the inset axes
# in the form of lower-left corner coordinates of inset axes, and its width and height respectively as a fraction of original axes
axins = ax.inset_axes([0.5, 0.5, 0.4, 0.4])

axins.plot([7,8,9],[2,8,6])

# set title to the inset axes
axins.set_title('Inset Plot')

axins1 = ax.inset_axes([0.3, 0.2, 0.4, 0.4])
axins1.plot([7,8,9],[2,8,6])
axins1.patch.set_alpha(0.8)
print(ax.get_xlim())
axins = ax.inset_axes([0.2, 0.2, 0.4, 0.4])
axins.plot([7,8,9],[2,8,6])
axins.patch.set_alpha(0.7)
# print the figure
plt.show(block = False)
plt.show()