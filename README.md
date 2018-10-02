# Compressed Sensing with Deep Image Prior

This repository provides code to reproduce results from the paper: [Compressed Sensing with Deep Image Prior and Learned Regularization](https://arxiv.org/pdf/1806.06438.pdf). Functionality for the learned regularization method is currently in progress.

Here are a few example results:

MNIST at 100 measurements | X-ray at 4000 measurements
--------------------------|---------------------------
<img src="https://github.com/davevanveen/compsensing_dip/blob/master/reconstructions/mnist/mnist_sample_100meas.png" alt="mnist_reconstr" width="300"> | <img src="https://github.com/davevanveen/compsensing_dip/blob/master/reconstructions/xray/xray_sample_4000meas.png" alt="mnist_reconstr" width="300">



#### MNIST reconstructions at 100 measurements
<img src="https://github.com/davevanveen/compsensing_dip/blob/master/reconstructions/mnist/mnist_sample_100meas.png" alt="mnist_reconstr" width="500">

#### X-ray reconstructions at 4000 measurements
<img src="https://github.com/davevanveen/compsensing_dip/blob/master/reconstructions/xray/xray_sample_4000meas.png" alt="mnist_reconstr" width="500">


### Preliminaries
---

1. Clone the repository
    ```shell
    $ git clone https://github.com/davevanveen/compsensing_dip.git
    $ cd compsensing_dip
    ```
    Please run all commands from the root directory of the repository, i.e from ```compsensing_dip/```

2. Install requirements
    ```shell
    $ pip install -r requirements.txt
    ```


### Plot reconstructions with existing data
---
1. Open jupyter notebook of plots
    ```shell
    $ jupyter notebook reconstructions.ipynb
    ```	
2. Set the variables ```DATASET``` and ```NUM_MEASUREMENTS_LIST```. Data is provided for NUM_MEASUREMENTS [10, 25, 50, 75, 100] on MNIST and [1000, 2000, 4000, and 8000] on the xray dataset.

3. Execute cells to view output.


### Generating new reconstructions on the MNIST or xray datasets
---
1. Execute the baseline command
	```shell
	$ python comp_sensing.py
	```
	which will run experiments with the default parameters specified in ```configs.json```

2. To generate reconstruction data according to user specified parameters, add command line arguments according to ```parser.py```. Example:
	```shell
	$ python comp_sensing.py --DATASET xray --NUM_MEASUREMENTS 2000 4000 8000 --BASIS csdip dct wavelet --NUM_ITER 500
	```

Note: To reduce runtime, redundant reconstruction data will not be generated for the same image at the same number of measurements. Thus if you wish to compare reconstructions while varying other parameters (e.g. learning rate, weight decay), your data must be manually relocated to avoid file overwrite.


### Running CS-DIP on a new dataset
---
1. Create a new directory ```/data/dataset_name/sub/``` which contains your images
2. In ```utils.py```, create a new DCGAN architecture. This will be similar to the pre-defined ```DCGAN_MNIST``` and ```DCGAN_XRAY``` but must have output dimension equal to the size of your new images. Output dimension can be changed by adjusting kernel_size, stride, and padding as discussed in the [torch.nn documentation](https://pytorch.org/docs/stable/nn.html#convtranspose2d). 
3. Update ```configs.json```, ```parser.py``` to set parameters for your dataset. Update ```cs_dip.py``` to import/initiate the corresponding DCGAN.
4. Generate and plot reconstructions according to instructions above.

Note: We recommend experimenting with the DCGAN architecture and dataset parameters to obtain the best possible reconstructions.




