# Official baseline code for the Third REACT Challenge (react2026)
[[Homepage]](https://sites.google.com/view/react2026/home)  [[Reference Paper (TBA)]]() [[Code]](https://github.com/reactmultimodalchallenge/baseline_react2026)

This repository provides baseline methods for the [Forth REACT Challenge](https://sites.google.com/view/react2026)

### Baseline paper:
- [https://github.com/reactmultimodalchallenge/baseline_react2026/blob/main/](https://github.com/reactmultimodalchallenge/baseline_react2026/blob/main/REACT_2026_Baseline%20(1).pdf)
### MARS dataset:
- Please send the signed EULA (https://github.com/reactmultimodalchallenge/baseline_react2026/blob/main/EULA_MARS%20dataset.pdf) to Dr Siyang Song at s.song@exeter.ac.uk 

### Challenge Description
Given the spatio-temporal behaviours expressed by a speaker at the time period, the proposed REACT 2025 Challenge will consist of the following two sub-challenges whose theoretical underpinnings have been defined and detailed in this paper.

#### Task 1 - Offline Appropriate Facial Reaction Generation
This task aims to develop a deep learning model that takes the entire speaker behaviour sequence as the input, and generates multiple appropriate and realistic / naturalistic spatio-temporal facial reactions, consisting of AUs, facial expressions, valence and arousal state representing the predicted facial reaction. As a result,  facial reactions are required to be generated for the task given each input speaker behaviour. 
#### Task 2 - Online Appropriate Facial Reaction Generation
This task aims to develop a deep learning model that estimates each frame, rather than taking all frames into consideration. The model is expected to gradually generate all facial reaction frames to form multiple appropriate and realistic / naturalistic spatio-temporal facial reactions consisting of AUs, facial expressions, valence and arousal state representing the predicted facial reaction. As a result,  facial reactions are required to be generated for the task given each input speaker behaviour. 

[//]: # (https://github.com/reactmultimodalchallenge/baseline_react2023/assets/35754447/8c7e7f92-d991-4741-80ec-a5112532460b)


## 🛠️ Dependency Installation

We provide detailed instructions for setting up the environment using conda. First, create and activate a new environment:
``` shell
conda create -n react python=3.10
conda activate react
```

### 1. Install PyTorch
First, check your CUDA version:
``` shell
nvidia-smi
```
Visit [Pytorch official website](https://pytorch.org/) to get the appropriate installation command. For example:
``` shell
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Install PyTorch3D Dependencies
Install the following dependencies:
``` shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```
For CUDA versions older than 11.7, you will need to install the CUB library. 
``` shell
conda install -c bottler nvidiacub
```

### 3. Install PyTorch3D
First, verify your CUDA version in Python:
``` shell
import torch
torch.version.cuda
```
[//]: # (Download `pytorch3d` file based on the version of python, cuda and pytorch from https://anaconda.org/pytorch3d/pytorch3d/files. For example, to install for Python 3.8, PyTorch 1.12.1 and CUDA 11.6, select the below file to download)
Download the appropriate `PyTorch3D` package from [Anaconda](https://anaconda.org/pytorch3d/pytorch3d/files) based on your Python, CUDA, and PyTorch versions. For example, for Python 3.10, CUDA 11.6, and PyTorch 1.12.0:

[//]: # (Finally install `pytorch3d` via the downloaded `.tar.bz2` file via conda)
``` shell
# linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
conda install linux-64_pytorch3d-0.7.5-py310_cu116_pyt1120.tar.bz2
```

### 4. Install Additional Dependencies
[//]: # (pip install omegaconf scikit-video pandas soundfile av decord tensorboard numpy tslearn scikit-image matplotlib imageio plotly opencv-python librosa einops)
Install all remaining dependencies specified in requirements.txt:
``` shell
pip install -r requirements.txt
```


## 👨‍🏫 Get Started 

<details><summary> <b> Data </b> </summary>
<p>
 
**Challenge Data Description ([Homepage](https://sites.google.com/cam.ac.uk/react2024)):**

We divided the datasets into training, test, and validation sets following an estimated 60%/20%/20% splitting ratio. Specifically, we split the datasets with a subject-independent strategy (i.e., the same subject was never included in the train and test sets).

[//]: # (- Dataset Directory Structure: &#40;training and validation sets are provided at this stage&#41;)
- *video-raw* folder contains raw videos (with the resolution of 1920 * 1080)
- *video-face-crop* folder contains face-cropped videos (with the resolution of 384 * 384)
- *facial-attributes* folder contains sequences of frame-level 25-dimension facial attributes (15 AUs’ occurrences, valence and arousal intensities, and the probabilities of eight categorical facial expressions)
- *coefficients* folder contains sequences of 58-dimension (52-d expression, 3-d rotation, and 3-d translation) 3DMM coefficients extracted from corresponding videos
- *audio* folder contains wav files extracted from raw video files

Appropriate real facial reactions (Ground-Truths):
- During data recording, the semantic contexts are carefully controlled through the 23 distinct sessions (session0, session1, …, session22), each of which is guided by a few pre-defined sentences posted by the speaker. This provides a consistent session-specific context across dyadic interactions between different speakers and listeners. More specifically, for the speaker behaviour expressed in a specific session, we define all facial reactions expressed by different listeners under the same session to be appropriate facial reactions (i.e., ground-truth) for responding to it.
   
**Data organization (`./data`) is listed below:**
The example of data structure.
```

├── val
├── test
├── train
    ├── coefficients (.npy)
    ├── video-face-crop (.mp4)
    ├── video-raw (.mp4)
        ├── speaker
            ├── session0
                ├── Camera-2024-06-21-103121-103102.mp4
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.mp4
                ├── ...
        ├── listener
            ├── session0
                ├── Camera-2024-06-21-103121-103102.mp4
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.mp4
                ├── ...
    ├── facial-attributes (.npy)
        ├── speaker
            ├── session0
                ├── Camera-2024-06-21-103121-103102.npy
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.npy
                ├── ...
        ├── listener
            ├── session0
                ├── Camera-2024-06-21-103121-103102.npy
                ├── ...
            ├── ...
            ├── session22
                ├── Camera-2024-07-17-104338-104241.npy
                ├── ...
    ├── audio (.wav)
        ├── speaker
            ├── session0
                ├── Camera-2024-06-21-103121-103102.wav
                ├── ...
            ├── ...
        ├── listener
            ├── session0
                ├── Camera-2024-06-21-103121-103102.wav
                ├── ...
            ├── ...
```

</p>
</details>

<details><summary> <b> External Tool Preparation </b> </summary>
<p>

We use 3DMM coefficients to represent a 3D listener or speaker, and for further 3D-to-2D frame rendering. The baselines leverage [3DMM model](https://github.com/LizhenWangT/FaceVerse) to extract 3DMM coefficients, and render 3D facial reactions.  

- You should first download 3DMM (FaceVerse version 2 model) at this [page](https://github.com/LizhenWangT/FaceVerse) 
 
  and then put it in the folder (`external/FaceVerse/data/`).
 
  We provide our extracted 3DMM coefficients (which are used for our baseline visualisation) at [OneDrive](https://drive.google.com/drive/folders/1RrTytDkkq520qUUAjTuNdmS6tCHQnqFu). 

  We also provide the `mean_face.npy` at this [OneDrive link](https://1drv.ms/u/c/4c787027becb2e91/EXhSObCHXUhHg0-Geyy4_6QB7b611XFgbJcIoGymcmzS-Q?e=NT8IKj) and `std_face.npy` at this [OneDrive link](https://1drv.ms/u/c/4c787027becb2e91/EdyIBxX-IlVEivdFxURn-BMBiK6JFSAXcp3qwCPNVboifQ?e=o5NgqM) and `reference_full.npy` at this [Onedrive link](https://1drv.ms/u/c/4c787027becb2e91/ERoBr5MNudxBgImW4jPt39sBwqFNSsvwX3OihUfU_TYpqw?e=h8mOqp) for 3DMM coefficients Data Normalization. Please download and put them in the folder (`external/FaceVerse/`).

[//]: # ( and reference_full )

Then, we use a 3D-to-2D tool [PIRender](https://github.com/RenYurui/PIRender) to render final 2D facial reaction frames.
 
- We re-trained the PIRender, and the well-trained model is provided at the [checkpoint](https://1drv.ms/u/c/4c787027becb2e91/EclM8oNvDeBKgI4I2lO95zkBXbTgRxuyGerKJ_EhYBuEtA?e=40O0Wc). Please put it in the folder (`external/PIRender/`).

[//]: # (https://1drv.ms/u/c/4c787027becb2e91/ERLUL_QTBABHoLzCTCbUZF8Bu6e_5o0YX31rA12yv0DIcQ?e=mWKgcn)

Finally, please download the compressed folder named `pretrained_models` from [this link](https://1drv.ms/u/c/4c787027becb2e91/EZ_l_EhvDbFOnmA_n69F1z0BpSqZumEcevc-iC3wVOhqhA?e=FlqhFb), and extract it into the project root directory.

</p>
</details>


<details><summary> <b> Training </b>  </summary>
<p>

<b>Generic online: </b>
<p>

<b>1. PerFRDiff + EEG</b>
<p>

```shell
python main.py \
    --config-name generic_online/motion_diffusion \
    trainer.batch_size=8 \
    stage=fit \
    data_dir=./datasets/REACT2026/ \
    trainer.model.diff_model.eeg_head.enabled=true \
    trainer.generic.train_eeg_head_only=false
```

<b>2. TransVAE + EEG</b>

```shell
python main.py \
    --config-name generic_online/motion_transvae \
    trainer.batch_size=2 \
    trainer.max_seq_len=256 \
    trainer.window_size=16 \
    stage=fit \
    data_dir=./datasets/REACT2026/ \
    trainer.train_eeg_head_only=false \
    trainer.model.eeg_head.enabled=true \
```


<b>Personalized online: </b>
<p>

<b>PerFRDiff rewrite-weight + EEG</b>
<p>

(a) Condition Input: Listener historical facial behaviours
```shell
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    stage=fit \
    data_dir=./datasets/REACT2026/ \
    trainer.generic.train_eeg=true \
    trainer.generic.train_eeg_head_only=false \
    trainer.main_model.args.personal_condition_mode=3dmm_only
```

(b) Condition Input: Personality_only
```shell
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    stage=fit \
    data_dir=./datasets/REACT2026/ \
    trainer.generic.train_eeg=true \
    trainer.generic.train_eeg_head_only=false \
    trainer.main_model.args.personal_condition_mode=personality_only
```

(c) Condition Input: Listener historical facial behaviours + Personality_only
```shell
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    stage=fit \
    data_dir=./datasets/REACT2026/ \
    trainer.generic.train_eeg=true \
    trainer.generic.train_eeg_head_only=false \
    trainer.main_model.args.personal_condition_mode=3dmm_personality
```

<b>Generic offline: </b>
<p>

<b> 1. TransVAE + EEG</b>
<p>

```shell
python main.py \
    --config-name generic_offline/motion_transvae \
    trainer.batch_size=4 \
    trainer.max_seq_len=750 \
    trainer.window_size=8 \
    stage=fit \
    data_dir=./datasets/REACT2026/ \
    trainer.train_eeg_head_only=false \
    trainer.model.eeg_head.enabled=true \
```

<b>2. ReGNN + EEG</b>
<p>

(a) Run this command from the `regnn/` directory:
```shell
cd ./regnn
```

(b) Extract the image features using the pre-trained swin_transformer (pretrained weights already provided in `./pretrained_models`):
```shell
python feature_extraction.py
```
(c) Train the REGNN by running the following shell:
```shell
python train.py \
    --logs-dir "Gmm-logs-eeg-head" \
    --data-dir ./datasets/REACT2026/ \
    --enable-eeg-head \
    --eeg-loss-weight 0.25 \
    --lr 0.0001 \
    --gamma 0.1 \
    --warmup-factor 0.01 \
    --milestones 9 \
    --batch-size 64 \
    --layers 2 \
    --act "ELU" \
    --seed 1 \
    --train-iters 100 \
    --norm \
    --neighbor-pattern "all" \
    --convert-type "direct" \
    --loss-mid
```
 
</p>
</details>

<details><summary> <b> Pretrained weights </b>  </summary>

- [ ] to be released

</details>

<details><summary> <b> Evaluation </b>  </summary>

[//]: # (- [ ] to be released)
For evaluation, please refer to `test` function in _./trainer/motion_diffusion.py_ (PerFRDiff baseline) or _./trainer/motion_transvae.py_ (Trans-VAE baseline). The metric computations are implemented in _./framework/utils/compute_metrics.py_. The validation set can be treated as the test set by loading it via the provided dataloader file. As in the baseline paper, all facial reactions from different participants within the same session are defined as ground-truths.
The pretrained model weights will be released soon.

<b>Generic online: </b>
<p>

<b>1. PerFRDiff + EEG</b>
<p>

```shell
python main.py \
    --config-name generic_online/motion_diffusion \
    trainer.batch_size=1 \
    stage=test \
    data_dir=./datasets/REACT2026/ \
    resume_id=<train-experiment-id> \
    trainer.generic.eval_eeg=true \
    trainer.model.diff_model.eeg_head.enabled=true
```

<b>2. TransVAE + EEG</b>

```shell
python main.py \
    --config-name generic_online/motion_transvae \
    trainer.batch_size=1 \
    trainer.max_seq_len=256 \
    trainer.window_size=16 \
    stage=test \
    data_dir=./datasets/REACT2026/ \
    trainer.data_transform=zero_center \
    resume_id=<train-experiment-id>  \
    trainer.eval_eeg=true \
    trainer.eval_eeg_metrics=true \
    trainer.eval_facial_metrics=true \
    trainer.save_results=true \
    trainer.renderer.do_render=false
```

<b>Personalized online: </b>
<p>

<b>PerFRDiff rewrite-weight + EEG</b>
<p>

(a) Condition Input: Listener historical facial behaviours
```shell
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    trainer.batch_size=1 \
    stage=test \
    data_dir=./datasets/REACT2026/ \
    resume_id=<train-experiment-id> \
    trainer.generic.eval_eeg=true \
    trainer.main_model.args.personal_condition_mode=3dmm_only
```

(b) Condition Input: Personality_only
```shell
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    trainer.batch_size=1 \
    stage=test \
    data_dir=./datasets/REACT2026/ \
    resume_id=<train-experiment-id> \
    trainer.generic.eval_eeg=true \
    trainer.main_model.args.personal_condition_mode=personality_only
```

(c) Condition Input: Listener historical facial behaviours + Personality_only
```shell
python main.py \
    --config-name personalized_online/perfrdiff_rewrite_weight \
    trainer.batch_size=1 \
    stage=test \
    data_dir=./datasets/REACT2026/ \
    resume_id=<train-experiment-id> \
    trainer.generic.eval_eeg=true \
    trainer.main_model.args.personal_condition_mode=3dmm_personality
```

<b>Generic offline: </b>
<p>

<b>1. TransVAE + EEG</b>
<p>

```shell
python main.py \
    --config-name generic_offline/motion_transvae \
    stage=test \
    data_dir=./datasets/REACT2026/ \
    trainer.batch_size=1 \
    trainer.max_seq_len=750 \
    trainer.window_size=8 \
    trainer.data_transform=zero_center \
    resume_id=<train-experiment-id> \
    trainer.eval_eeg=true \
    trainer.eval_eeg_metrics=true \
    trainer.eval_facial_metrics=true \
    trainer.save_results=true \
    trainer.renderer.do_render=false
```

<b>2. ReGNN + EEG</b>

```shell
python train.py \
  --test \
  --logs-dir "Gmm-logs-eeg-head" \
  --data-dir "./datasets/REACT2026/" \
  --model-pth "./baseline_react2026-main2/regnn/Gmm-logs-eeg-head/mhp-eeg-head-last-seed1.pth" \
  --enable-eeg-head \
  --eval-eeg \
  --metric-threads 1 \
  --eval-clip-batch-size 1 \
  --layers 2 \
  --act "ELU" \
  --seed 1 \
  --norm \
  --neighbor-pattern "all" \
  --convert-type "direct"
```


</details>


## 🖊️ Citation

### Submissions should cite the following papers:

#### Theory paper and baseline paper:

[1] Song, Siyang, Micol Spitale, Yiming Luo, Batuhan Bal, and Hatice Gunes. "Multiple Appropriate Facial Reaction Generation in Dyadic Interaction Settings: What, Why and How?." arXiv preprint arXiv:2302.06514 (2023).

[2] Song, Siyang, Micol Spitale, Xiangyu Kong, Hengde Zhu, Cheng Luo, Cristina Palmero, German Barquero et al. "React 2025: the third multiple appropriate facial reaction generation challenge." In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 13979-13984. 2025.

[3] Song, Siyang, Micol Spitale, Cheng Luo, Cristina Palmero, German Barquero, Hengde Zhu, Sergio Escalera et al. "React 2024: the second multiple appropriate facial reaction generation challenge." In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), pp. 1-5. IEEE, 2024. 

[4] Song, Siyang, Micol Spitale, Cheng Luo, Germán Barquero, Cristina Palmero, Sergio Escalera, Michel Valstar et al. "REACT2023: The First Multiple Appropriate Facial Reaction Generation Challenge." In Proceedings of the 31st ACM International Conference on Multimedia, pp. 9620-9624. 2023.


#### Annotation, basic feature extraction tools and baselines:

[6] Song, Siyang, Yuxin Song, Cheng Luo, Zhiyuan Song, Selim Kuzucu, Xi Jia, Zhijiang Guo, Weicheng Xie, Linlin Shen, and Hatice Gunes. "GRATIS: Deep Learning Graph Representation with Task-specific Topology and Multi-dimensional Edge Features." arXiv preprint arXiv:2211.12482 (2022).

[7] Luo, Cheng, Siyang Song, Weicheng Xie, Linlin Shen, and Hatice Gunes. (2022, July) "Learning multi-dimensional edge feature-based au relation graph for facial action unit recognition." Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (pp. 1239-1246).

[8] Toisoul, Antoine, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, and Maja Pantic. "Estimation of continuous valence and arousal levels from faces in naturalistic conditions." Nature Machine Intelligence 3, no. 1 (2021): 42-50.

[9] Eyben, Florian, Martin Wöllmer, and Björn Schuller. "Opensmile: the munich versatile and fast open-source audio feature extractor." In Proceedings of the 18th ACM international conference on Multimedia, pp. 1459-1462. 2010.

### Submissions are encouraged to cite previous personalized facial reaction generation papers:

[10] Zhu, Hengde, Xiangyu Kong, Weicheng Xie, Xin Huang, Linlin Shen, Lu Liu, Hatice Gunes, and Siyang Song. "Perfrdiff: Personalised weight editing for multiple appropriate facial reaction generation." In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 9495-9504. 2024.

[11] Zhu, Hengde, Xiangyu Kong, Weicheng Xie, Xin Huang, Xilin He, Lu Liu, Linlin Shen, Wei Zhang, Hatice Gunes, and Siyang Song. "PerReactor: Offline Personalised Multiple Appropriate Facial Reaction Generation." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 2, pp. 1665-1673. 2025.

[12] Song, Siyang, Zilong Shao, Shashank Jaiswal, Linlin Shen, Michel Valstar, and Hatice Gunes. "Learning Person-specific Cognition from Facial Reactions for Automatic Personality Recognition." IEEE Transactions on Affective Computing (2022).

[13] Shao, Zilong, Siyang Song, Shashank Jaiswal, Linlin Shen, Michel Valstar, and Hatice Gunes. "Personality recognition by modelling person-specific cognitive processes using graph representation." In proceedings of the 29th ACM international conference on multimedia, pp. 357-366. 2021.



### Submissions are encouraged to cite previous generic facial reaction generation papers:

[14] Huang, Yuchi, and Saad M. Khan. "Dyadgan: Generating facial expressions in dyadic interactions." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 11-18. 2017.

[15] Huang, Yuchi, and Saad Khan. "A generative approach for dynamically varying photorealistic facial expressions in human-agent interactions." In Proceedings of the 20th ACM International Conference on Multimodal Interaction, pp. 437-445. 2018.

[16] Barquero, German, Sergio Escalera, and Cristina Palmero. "Belfusion: Latent diffusion for behavior-driven human motion prediction." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2317-2327. 2023.

[17] Zhou, Mohan, Yalong Bai, Wei Zhang, Ting Yao, Tiejun Zhao, and Tao Mei. "Responsive listening head generation: a benchmark dataset and baseline." In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVIII, pp. 124-142. Cham: Springer Nature Switzerland, 2022.

[18] Luo, Cheng, Siyang Song, Weicheng Xie, Micol Spitale, Zongyuan Ge, Linlin Shen, and Hatice Gunes. "Reactface: Online multiple appropriate facial reaction generation in dyadic interactions." IEEE Transactions on Visualization and Computer Graphics 31, no. 9 (2024): 6190-6207.

[19] Xu, Tong, Micol Spitale, Hao Tang, Lu Liu, Hatice Gunes, and Siyang Song. "Reversible graph neural network-based reaction distribution learning for multiple appropriate facial reactions generation." IEEE Transactions on Affective Computing (2026).

[20] Liang, Cong, Jiahe Wang, Haofan Zhang, Bing Tang, Junshan Huang, Shangfei Wang, and Xiaoping Chen. "Unifarn: Unified transformer for facial reaction generation." In Proceedings of the 31st ACM International Conference on Multimedia, pp. 9506-9510. 2023.

[21] Yu, Jun, Ji Zhao, Guochen Xie, Fengxin Chen, Ye Yu, Liang Peng, Minglei Li, and Zonghong Dai. "Leveraging the latent diffusion models for offline facial multiple appropriate reactions generation." In Proceedings of the 31st ACM International Conference on Multimedia, pp. 9561-9565. 2023.

[22] Hoque, Ximi, Adamay Mann, Gulshan Sharma, and Abhinav Dhall. "BEAMER: Behavioral Encoder to Generate Multiple Appropriate Facial Reactions." In Proceedings of the 31st ACM International Conference on Multimedia, pp. 9536-9540. 2023.

[23] Nguyen, Dang-Khanh, Prabesh Paudel, Seung-Won Kim, Ji-Eun Shin, Soo-Hyung Kim, and Hyung-Jeong Yang. "Multiple facial reaction generation using gaussian mixture of models and multimodal bottleneck transformer." In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), pp. 1-5. IEEE, 2024.

[24] Hu, Guanyu, Jie Wei, Siyang Song, Dimitrios Kollias, Xinyu Yang, Zhonglin Sun, and Odysseus Kaloidas. "Robust facial reactions generation: An emotion-aware framework with modality compensation." In 2024 IEEE International Joint Conference on Biometrics (IJCB), pp. 1-10. IEEE, 2024.

[25] Liu, Zhenjie, Cong Liang, Jiahe Wang, Haofan Zhang, Yadong Liu, Caichao Zhang, Jialin Gui, and Shangfei Wang. "One-to-many appropriate reaction mapping modeling with discrete latent variable." In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), pp. 1-5. IEEE, 2024.

[26] Dam, Quang Tien, Tri Tung Nguyen Nguyen, Dinh Tuan Tran, and Joo-Ho Lee. "Finite scalar quantization as facial tokenizer for dyadic reaction generation." In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), pp. 1-5. IEEE, 2024.

[27] Luo, Jiachen, Jiajun He, Shuai Shen, Lin Wang, Huy Phan, Joshua Reiss, Lin Haijun, Bjoern Schuller, Zeyu Fu, and Siyang Song. "MReactor: Offline Multiple Appropriate Facial Reaction Generation with Hierarchical Cognitive Disentanglement." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3354-3363. 2026.

[28] Xie, Weicheng, Chunlin Yan, Siyang Song, Zitong Yu, Linlin Shen, and Laizhong Cui. "Smooth Online Multiple Appropriate Facial Reaction Generation." In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 5804-5813. 2025.

[29] Mao, Qirong, Qiwei Wu, Na Liu, Yakui Ding, and Lijian Gao. "Scattering-Conditioned Diffusion Models for Multiple Appropriate Facial Reaction Generation." In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 13985-13991. 2025.

[30] Wang, Peng, Pujun Xue, Xiaofeng Liu, and Tongjuan Ji. "Explaining Listener Reactions: Personality-Guided Facial Response Generation with Cross-Modal Attention." In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 13997-14003. 2025.

[31] Huang, Jiajian, and Zitong Yu. "Multiple Appropriate Facial Reaction Generation Based on Multi-View Transformation of Speaker Video." In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 13992-13996. 2025.

[32] Nguyen, Minh-Duc, Hyung-Jeong Yang, Ngoc-Huynh Ho, Soo-Hyung Kim, Seungwon Kim, and Ji-Eun Shin. "Vector quantized diffusion models for multiple appropriate reactions generation." In 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), pp. 1-5. IEEE, 2024.

[33] Lv, Qincheng, Xiaofeng Liu, Jie Li, Rongrong Ni, Pujun Xue, and Siyang Song. "Hierarchical multimodal decoupling-fusion framework for offline multiple appropriate facial reaction generation." In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1-5. IEEE, 2025.

[34] Luo, Cheng, Siyang Song, Siyuan Yan, Zhen Yu, and Zongyuan Ge. "ReactDiff: Fundamental Multiple Appropriate Facial Reaction Diffusion Model." In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 5607-5616. 2025.

[35] Li, Jiaming, Sheng Wang, Xin Wang, Yitao Zhu, Honglin Xiong, Zixu Zhuang, and Qian Wang. "Reactdiff: Latent diffusion for facial reaction generation." Neural Networks 189 (2025): 107596.

  

## 🤝 Acknowledgement
Thanks to the open source of the following projects:

- [FaceVerse](https://github.com/LizhenWangT/FaceVerse) &#8194;

- [PIRender](https://github.com/RenYurui/PIRender) &#8194;

[//]: # (<details><summary> <b> Validation </b>  </summary>)

[//]: # (<p>)

[//]: # ( Follow this to evaluate Trans-VAE or BeLFusion after training, or downloading the pretrained weights.)

[//]: # ( )
[//]: # (- Before validation, run the following script to get the martix &#40;defining appropriate neighbours in val set&#41;:)

[//]: # ( ```shell)

[//]: # ( cd tool)

[//]: # ( python matrix_split.py --dataset-path ./data --partition val)

[//]: # ( ```)

[//]: # (&nbsp;  Please put files &#40;`data_indices.csv`, `Approprirate_facial_reaction.npy` and `val.csv`&#41; in the folder `./data/`.)

[//]: # (  )
[//]: # (- Then, evaluate a trained model on val set and run:)

[//]: # ()
[//]: # ( ```shell)

[//]: # (python evaluate.py  --resume ./results/train_offline/best_checkpoint.pth  --gpu-ids 1  --outdir results/val_offline --split val)

[//]: # (```)

[//]: # ( )
[//]: # (&nbsp; or)

[//]: # ( )
[//]: # (```shell)

[//]: # (python evaluate.py  --resume ./results/train_online/best_checkpoint.pth  --gpu-ids 1  --online --outdir results/val_online --split val)

[//]: # (```)

[//]: # ( )
[//]: # (- For computing FID &#40;FRRea&#41;, run the following script:)

[//]: # ()
[//]: # (```)

[//]: # (python -m pytorch_fid  ./results/val_offline/fid/real  ./results/val_offline/fid/fake)

[//]: # (```)

[//]: # (</p>)

[//]: # (</details>)


[//]: # (<details><summary> <b> Other baselines </b>  </summary>)

[//]: # (<p>)

[//]: # ( )
[//]: # (- Run the following script to sequentially evaluate the naive baselines presented in the paper:)

[//]: # ( ```shell)

[//]: # ( python run_baselines.py --split SPLIT)

[//]: # ( ```)

[//]: # ( SPLIT can be `val` or `test`.)

[//]: # (</p>)

[//]: # (</details>)


[//]: # (<details><summary> <b> Pretrained weights </b>  </summary>)

[//]: # ( If you would rather skip training, download the following checkpoints and put them inside the folder './results'.)

[//]: # (<p>)

[//]: # ( )
[//]: # ( <b>Trans-VAE</b>: TBA)

[//]: # ( )
[//]: # ( <b>BeLFusion</b>: [download]&#40;https://ubarcelona-my.sharepoint.com/:f:/g/personal/germanbarquero_ub_edu/EkRisY7MzX5MnP6tIVYhkdYBInl3lw3XXJuW6fEXnij4aQ?e=XZHvSw&#41;)

[//]: # ()
[//]: # ( <b>REGNN</b>: [download]&#40;https://drive.google.com/drive/folders/18I-yfpY1mlLqp4-E443xxwXNWh3ET-RN?usp=sharing&#41;)

[//]: # ( )
[//]: # (</details>)

[//]: # ()
[//]: # (<details><summary> <b> Evaluation </b>  </summary>)

[//]: # (<p>)

[//]: # ( Follow this to evaluate Trans-VAE or BeLFusion after training, or downloading the pretrained weights.)

[//]: # ( )
[//]: # (- Before testing, run the following script to get the martix &#40;defining appropriate neighbours in test set&#41;:)

[//]: # ( ```shell)

[//]: # ( cd tool)

[//]: # ( python matrix_split.py --dataset-path ./data --partition test)

[//]: # ( ```)

[//]: # (&nbsp;  Please put files &#40;`data_indices.csv`, `Approprirate_facial_reaction.npy` and `test.csv`&#41; in the folder `./data/`.)

[//]: # (  )
[//]: # (- Then, evaluate a trained model on test set and run:)

[//]: # ()
[//]: # ( ```shell)

[//]: # (python evaluate.py  --resume ./results/train_offline/best_checkpoint.pth  --gpu-ids 1  --outdir results/test_offline --split test)

[//]: # (```)

[//]: # ( )
[//]: # (&nbsp; or)

[//]: # ( )
[//]: # (```shell)

[//]: # (python evaluate.py  --resume ./results/train_online/best_checkpoint.pth  --gpu-ids 1  --online --outdir results/test_online --split test)

[//]: # (```)

[//]: # ()
[//]: # ( )
[//]: # (- For computing FID &#40;FRRea&#41;, run the following script:)

[//]: # ()
[//]: # (```)

[//]: # (python -m pytorch_fid  ./results/test_offline/fid/real  ./results/test_offline/fid/fake)

[//]: # (```)

[//]: # ()
[//]: # ( For evaluation of REGNN, there are two steps.)

[//]: # ( - First generate facial reactions and save them by running the script within the folder `regnn`:)

[//]: # ( ```)

[//]: # ( bash scripts/inference.sh)

[//]: # ( ```)

[//]: # ( - Then evaluate the predicted facial reactions by running the `evaluation.py` in the folder `regnn`:)

[//]: # ( ```)

[//]: # ( python evaluation.py --data-dir <data-dir> --pred-dir <pred-dir> split test)

[//]: # ( ```)

[//]: # (</p>)

[//]: # (</details>)
