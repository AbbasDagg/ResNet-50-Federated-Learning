# ResNet-50-Federated-Learning
### running using wsl
if you are running using a gpu install cuda on your wsl machine the official steps could be found [here](https://developer.nvidia.com/cuda-13-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) Note: check if 13.0 is supported for your gpu.  
Next clone the repo using:
```bash
git clone git@github.com:AbbasDagg/ResNet-50-Federated-Learning.git
```
then install the requirements using:
```bash
pip install -r requirements.txt
```
**Note** if you are running with cuda maybe you need to install torch with cuda check the oficial [wibsite](https://pytorch.org/get-started/locally/) for more info.

To run the main code run the main file [fl_job.py](federated_learning/fl_job.py)
```bash
python federated_learning/fl_job.py
```
