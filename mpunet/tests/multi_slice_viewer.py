# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from aicsimageio import AICSImage
import sys, os
import numpy as np
import czifile
from skimage import io
import matplotlib.pyplot as plt
import numpy
#from mayavi.mlab import *
import nibabel as nib
import matplotlib.pyplot as plt

def show_center(data, last_dimension=0):
    if len(data.shape) == 4:
        m,n,o,p = data.shape
        slice_0 = data[m//2, :, :, last_dimension]
        slice_1 = data[:, n//2, :, last_dimension]
        slice_2 = data[:, :, o//2, last_dimension]
        show_slices([slice_0, slice_1, slice_2])
    elif len(data.shape) == 3:
        m,n,o = data.shape
        slice_0 = data[m//2, :, :]
        slice_1 = data[:, n//2, :]
        slice_2 = data[:, :, o//2]

        show_slices([slice_0, slice_1, slice_2])

def get_certain_slice(data, dimension=2,position=None, last_dimension=0):
    if len(data.shape) == 4:
        data = data[:,:,:,last_dimension]
    m,n,o = data.shape
    if dimension == 0:
        slice = data[position or m//2, :, :]
    elif dimension == 1:
        slice = data[:, position or n//2, :]
    else:
        slice = data[:, :, position or o//2]

    return slice


def show_slices(slices,figsize=(20,20)):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices),figsize=figsize)
    for i, slice in enumerate(slices):
         axes[i].imshow(slice.T, cmap="gray", origin="lower")


def load_and_to_array(example_filename):
    img = nib.load(example_filename)
    data = img.get_fdata()
    data.shape
    return data

class multi_show():
    def __init__(self, volume, stride=10):
        self.volume = volume
        if len(volume.shape) == 4:
            self.volume = self.volume[:,:,:,0]
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
        self.ax.imshow(self.volume[self.ax.index], cmap='Greys')
        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def process_key(self, event):
        self.fig = event.canvas.figure
        self.ax = self.fig.axes[0]
        #print('you pressed', event.key, event.xdata, event.ydata)

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
