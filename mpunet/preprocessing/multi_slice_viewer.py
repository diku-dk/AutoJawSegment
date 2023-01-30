# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np


class multi_show():
    def __init__(self, volume, stride=10, show_dim=0):
        if show_dim != 0:
            volume = np.moveaxis(volume, show_dim, 0)
        self.volume = volume
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.stride = stride

    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def multi_slice_viewer(self):
        # remove_keymap_conflicts({'j', 'k'})

        self.ax.volume = self.volume
        self.ax.index = self.volume.shape[0] // 2
        self.ax.imshow(self.volume[self.ax.index], cmap='gray')
        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def process_key(self, event):
        self.fig = event.canvas.figure
        self.ax = self.fig.axes[0]
        # print('you pressed', event.key, event.xdata, event.ydata)

        print(self.ax.index)
        print(event.key)

        if event.key in ['j', 'up']:
            print('you pressed j')
            self.previous_slice(self.stride)
        elif event.key in ['k', 'down']:
            print('you pressed k')
            self.next_slice(self.stride)
        self.fig.canvas.draw()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.previous_slice(1)
        else:
            self.next_slice(1)
        self.fig.canvas.draw()

    def previous_slice(self, stride_per_step=10):
        volume = self.volume
        self.ax.index = (self.ax.index - stride_per_step) % volume.shape[0]  # wrap around using %
        self.ax.images[0].set_array(self.volume[self.ax.index])

    def next_slice(self, stride_per_step=10):
        volume = self.ax.volume
        self.ax.index = (self.ax.index + stride_per_step) % volume.shape[0]  # wrap around using %
        self.ax.images[0].set_array(self.volume[self.ax.index])

    def matplot3D(self):
        self.multi_slice_viewer()
        plt.show()