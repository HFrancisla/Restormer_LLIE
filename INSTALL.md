# 30系

conda create -n res37 python=3.7 -y

# 首选pip

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f <https://download.pytorch.org/whl/torch_stable.html>

# conda不太行感觉

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c nvidia -c pytorch -c conda-forge -y

# 40系

conda create -n res39 python=3.9 -y

# 首选pip

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url <https://download.pytorch.org/whl/cu118>

# conda不太行感觉

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Next

pip install matplotlib scikit-learn scikit-image "opencv-python<4.10.0" yacs joblib natsort h5py tqdm wandb

pip install einops gdown addict future lmdb "numpy<2" pyyaml requests scipy tb-nightly yapf lpips

# if needed

pip install "setuptools<70.0.0"

python setup.py develop --no_cuda_ext

# 或

pip install -v -e . --no-build-isolation --config-settings="--global-option=--no_cuda_ext"

pip uninstall pillow

# pillow 9.5(这里应该是大小写的问题，重新安装就好了)

pip install pillow

# python3.9

pip install "numpy<2" "opencv-python<4.9"

# 如遇到lmdb有问题，卸载后使用conda安装编译好的lmdb二进制文件

pip uninstall lmdb -y
conda install -c conda-forge python-lmdb

python -m basicsr.train -opt LLIE\Options\Restormer_128_2_60k.yml
