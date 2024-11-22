
The default installation guideline results in severe dependency errors. 

Downgrade from numpy 2 . Updated requirements.txt

Downgrade ml_dtypes to 0.2.0 


Downgrade scipyto 1.12 (https://github.com/octo-models/octo/issues/71)

Downgrade tensorflow_probability to 0.17 ... 

Downgrade wandb to 0.12.14

Downgrade protobuf as well to 3.20.*


Stll the error :
```
I1121 18:12:41.117942 140334031860352 xla_bridge.py:450] Unable to initialize backend 'plugin': xla_extension has no attributes named get_plugin_device_client. Compile TensorFlow with //tensorflow/compiler/xla/python:enable_plugin_device set to true (defaults to false) to enable this.
/home/baburamshapure/miniconda3/envs/rlpd/lib/python3.9/site-packages/jax/_src/interpreters/pxla.py:186: DeprecationWarning: ml_dtypes.float8_e4m3b11 is deprecated. Use ml_dtypes.float8_e4m3b11fnuz
  return xc.batched_device_put(aval, sharding, xs, devices, committed)  # type: ignore
2024-11-21 18:12:41.236285: E external/xla/xla/stream_executor/cuda/cuda_dnn.cc:439] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
2024-11-21 18:12:41.236328: E external/xla/xla/stream_executor/cuda/cuda_dnn.cc:443] Memory usage: 2571698176 bytes free, 12626493440 bytes total.
Traceback (most recent call last):
  File "/media/baburamshapure/New Volume/GEEK/causality/rlpd/train_finetuning.py", line 237, in <module>

```


Successfully installed jax-0.4.9 jaxlib-0.4.9+cuda12.cudnn88

Seems to be some problem with the cudnn version. Keeps intalling cudnn 9... the older libraries I think need 8.8.y.z

```
conda activate rlpd && 
python3 -m pip install nvidia-cudnn-cu12==8.8.1.3
```

Now its a bit messy. I was trying to update the cuda toolkit but seems like I royally messed up. .. 

Somehow the cuda toolkit got downgraded to 12.3 ... updated to 12.6 using following: 0

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```


**IT RUNS!!**


Next error : 
TypeError: Object of type ArrayImpl is not JSON serializable

seems like some wandb error ... jax is trying to log something wandb does not support .. 

Bump wandb to 0.13.8 following this: https://github.com/wandb/wandb/issues/4735


