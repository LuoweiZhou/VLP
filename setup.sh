echo "----------------- Check Env -----------------"
nvidia-smi
cat /usr/local/cuda/version.txt
nvcc -V
python -V
python -c 'import torch; print(torch.__version__)'

echo "----------------- Check File System -----------------"
echo "I am " $(whoami)
echo -n "CURRENT_DIRECTORY "
pwd

echo "----------------- Start Installing -----------------"
PWD_DIR=$(pwd)
cd $(mktemp -d)

echo "----------------- Install NCCL -----------------"
sudo apt -y update
sudo apt install vim zip unzip ca-certificates-java openjdk-8-jdk -y
# cp /mnt/pkg/nccl-repo-ubuntu1604-2.4.7-ga-cuda10.0_1-1_amd64.deb .
# sudo dpkg -i nccl-repo-ubuntu1604-2.4.7-ga-cuda10.0_1-1_amd64.deb
# sudo apt install libnccl2=2.4.7-1+cuda10.0 libnccl-dev=2.4.7-1+cuda10.0 -y

echo "----------------- Install Apex -----------------"
git clone -q https://github.com/NVIDIA/apex.git
cd apex
git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python setup.py install --cuda_ext --cpp_ext

echo "----------------- Start Training -----------------"
echo $(pwd)
echo "print java version"
java -version
export PYTHONPATH=$PWD_DIR/pythia:$PWD_DIR/pythia/pythia/legacy:$PWD_DIR:$PYTHONPATH
