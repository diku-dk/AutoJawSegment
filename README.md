#  Backbone model: Multi-Planar U-Net (modified)
Most of the repository is based on the official implemenation of original Multi-Planar U-Net.
Please follow https://github.com/perslev/MultiPlanarUNet for the general installation and usage of Multi-Planar U-Net

But make sure you clone from the current repository.

---

##  Installation
```
git clone https://github.com/diku-dk/AutoJawSegment
conda create -n MPUNet python==3.9
conda activate MPUNet
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install -e AutoJawSegment
```
####  test if gpu activated
 ```
 python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Initializing two projects, one for pretraining on inaccurate data, and one for fine-tuning as follows:

```
# Initialize a project for pretraining/or just training without a pretrained model
mp init_project --name my_pretraining_project --data_dir ./pretraining_data_folder
# Initialize a project for finetuing
mp init_project --name my_finetune_project --data_dir ./fineune_data_folder
```

The YAML file named ```train_hparams.yaml```, which stores all hyperparameters will also be 
created for the two projects respectively. Any 
parameter in this file may be specified manually, but can all be set 
automatically. Please follow original MPUNet for the data format

---
## Pre-training
The model can now be pre-trained as follows:

```
cd my_pretraining_project
mp train --overwrite --num_GPUs=1   # Any number of GPUs (or 0)  --force_GPU=0 # set this if gpu is doing other works from deep learning
```

## Fine-tuning
The standard fine-tuning process without disatance weight map can be applied by invoking 
```
# make sure you are at root dir
cd my_finetune_project
mp train --num_GPUs=1 --overwrite --transfer_last_layer  \
--force_GPU=0 --initialize_from '../my_pretraining_project/model/@epoch_xxx_val_dice_xxxx.h5'
```
Make sure you change ```xxx``` according to the .h5 file of the model generated from pre-training step

Also make sure to comment out ```transfer_last_layer``` if not working on the same class

## Predict
The trained model can now be evaluated on the testing data in 
```data_folder/test``` by invoking:

```
mp predict --num_GPUs=1 --force_GPU=0 --sum_fusion --overwrite --no_eval --by_radius --out_dir predictions 
```
note that the ```--sum_fusion``` tag is essential since we are not training 
another fusion model. ```--by_radius``` tage is also essential in ensuring different sampling strategy during training
and testing to avoid incomplete boundaries, as mentioned in the paper.

This will create a folder ```my_project/predictions``` storing the predicted 
images along with dice coefficient performance metrics.

## Post-Processing
Usually, each class should be a single connected component. Run the following to keep the largest component
for each class
```
python postprocessing/ccd
```

Other post-processing such as opening to remove small floating parts and closing to remove holes
are often also helpful, but would need to specify the size.


## Evaluate/Numerical Validation
The results can be evaluated on the testing data to get DICE, GapDICE, and Hausdoff Distance, and ASSD by running

```
python mpunet/evaluate/compute_metrics.py 
```

This will print the mean and standard deviation for the 4 metrics, as well as generate .csv files for the score for each image. 
You can use simple a script or Excel functions to get the mean and std of the overall performance
