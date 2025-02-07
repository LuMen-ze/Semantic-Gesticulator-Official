

# SIGGRAPH 2024 Journal Track<br /> Semantic Gesticulator: Semantics-Aware Co-Speech Gesture Synthesis

[Zeyi Zhang*](https://lumen-ze.github.io/), [Tenglong Ao*](https://aubrey-ao.github.io/), Yuyao Zhang*, [Qingzhe Gao](https://talegqz.github.io/), Chuan Lin, [Baoquan Chen](https://cfcs.pku.edu.cn/baoquan/), [Libin Liu†](http://libliu.info/)


***
<p align=center>
<img src='media/sg.jpg' width='90%'> </img>
</p>

<p align="center">
-
Video (<a href="https://www.youtube.com/watch?v=gKGqCE7id4U">YouTube</a>)
-
Paper (<a href="https://arxiv.org/abs/2405.09814">arXiv</a>)
-
Project Page (<a href="https://pku-mocca.github.io/Semantic-Gesticulator-Page/">github</a>)
-

</p>

This is a reimplemention of Semantic Gesticulator: Semantics-Aware Co-Speech Gesture Synthesis.

This codebase provides:
- [x] SeG dataset
- [x] pretrained models
- [x] training & inference codes

If you use the dataset or codes, please cite our [Paper](https://arxiv.org/abs/2405.09814)

```
@article{
  Zhang2024SemanticGesture,
  author = {Zhang, Zeyi and Ao, Tenglong and Zhang, Yuyao and Gao, Qingzhe and Lin, Chuan and Chen, Baoquan and Liu, Libin},
  title = {Semantic Gesticulator: Semantics-Aware Co-Speech Gesture Synthesis},
  journal = {ACM Trans. Graph.},
  issue_date = {July 2024},
  numpages = {17},
  doi = {10.1145/3658134},
  publisher = {ACM},
  address = {New York, NY, USA},
  keywords = {co-speech gesture synthesis, multi-modality, retrieval augmentation}
}
```

## SeG Dataset
> **Collection Period** of the semantic gesture inventory/list and the motion capture data: ```December 2023 - May 2024```.<br /> <br /> 
> **Copyright Source** of semantic gestures: the "Copyright Source" column in ```SeG_list.xlsx``` indicates the source of each semantic gesture, with the copyright belonging to the respective references.

The MOCCA Semantic Gesture Dataset (SeG) comprises 208 types of semantic gestures commonly utilized globally. Each record in the dataset includes a sequence of captured motions and associated meta-information.


As shown below, the meta-information includes semantics-aware index, label, description, contextual meaning, and example sentence, facilitating a thorough organization and analysis of human gestures. The related information is stored in ```./SeG_dataset/SeG_list.xlsx```.

<p align=center>
<img src='media/seg_example.jpeg' width='60%'> </img>
</p>

As for motion, we have open-sourced the motion capture data from a male performer, comprising 544 motion files in the .bvh format. Each gesture is represented, on average, in 2.6 distinct ways. These motion files are named using the format "label" + "represented index" and stored in ```./SeG_dataset/bvh```. You can also get these files in the .zip format from [Google Drive](https://drive.google.com/file/d/1qVQ4-XwAzhfoHRM6EzR9BrGI1lD1vqVW/view?usp=sharing).

## SG Code
Semantic-Gesticulator provides a framework for generating gestures from audio input using pre-trained models. This part includes tools for inference, data preparation, and training models.

---

### Table of Contents
- [Environment Setup](SG_code/README.md#environment-setup)
- [Pretrained Models](SG_code/README.md#pretrained-models)
- [Inference](SG_code/README.md#inference)
- [Training](SG_code/README.md#training)
  - [Data Preparation](SG_code/README.md#1-data-preparation)
  - [Training RVQ](SG_code/README.md#2-training-rvq)
  - [Training GPT](SG_code/README.md#3-training-gpt)
- [Citation](SG_code/README.md#Citation)
- [License](SG_code/README.md#license)

---

### Environment Setup

To create and activate the required Python environment:

```bash
conda create -n SG python=3.12.7 -y
conda activate SG
pip install -r requirements.txt
pip install -U openai-whisper
```

### Pretrained Models

Download the pretrained models for Gesture Generator (Residual VQ-VAE & GPT-2) from [google drive](https://drive.google.com/drive/folders/1zstSOaMJF5iAwLetJjspIPja0yVywle6?usp=sharing) and place them in the `SG_code/pretrained_models` directory.

And then download the pretrained models for Semantic Gesture Retriever from [hugging face](https://huggingface.co/illusence/Semantic_Gesture_Retrieval_Model) and place them under the `SG_code/retrieval_model` directory.

Please note that the retrieval model is a new version based on an open-source model, Qwen-2.5-7B, allowing users easier access. This version is different from the original version based on ChatGPT 3.5-turbo described in the paper. And as we noticed, the new version based on Qwen-2.5-7B is not as good as the original version based on ChatGPT 3.5-turbo mainly in the semantic retrieval accurancy and instruction following ability.

The directory structure should look like this:

- SG_code
  - pretrained_models
    - rqvae.pt
    - gpt0.pt
    - gpt1.pt
    - gpt2.pt
    - gpt3.pt
  - retrieval_model
    - model-00001-of-00004.safetensors
    - model-00002-of-00004.safetensors
    - model-00003-of-00004.safetensors
    - model-00004-of-00004.safetensors
    - ...


### Inference

#### 1. Generate Rhythmic Gestures (w/o Semantic Gestures) from Audio
To run inference and generate gestures from an audio file:

-	Ensure your audio file is in .wav format.

-	Fill in the appropriate audio_path (path to your audio file) and save_dir (output directory) in the command below.


```bash
cd SG_code

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 generate_gestures.py \
  --audio_path <your_audio_file_path> \
  --save_dir <your_save_dir> \
  --rqvae_path './pretrained_models/rqvae.pt' \
  --model_path_0 './pretrained_models/gpt_0.pt' \
  --model_path_1 './pretrained_models/gpt_1.pt' \
  --model_path_2 './pretrained_models/gpt_2.pt' \
  --model_path_3 './pretrained_models/gpt_3.pt' \
  --init_body_pose_code 128 \
  --init_hands_pose_code 258 \
  --processed_dataset_dir './Data/SG_processed'
  ```


#### 2. Generate Semantics-Enhanced Gestures from Audio
To generate semantics-enhanced gestures from an audio file, you need to deploy the API service first. Please set the CUDA visible devices and start the service using the script in your terminal:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <the global path to the retrieval model folder> --port 8000 --tensor-parallel-size 4 --gpu_memory_utilization 0.7
```

This command will start the API service on port `8000` using **at least 4 3090-GPUs** with 70% memory utilization. 

And then, you can generate semantic gestures from an audio file by running the following command:

```bash
cd SG_code

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 generate_semantic_gestures.py \
  --audio_path <your_audio_file_path> \
  --save_dir <your_save_dir> \
  --rqvae_path './pretrained_models/rqvae.pt' \
  --model_path_0 './pretrained_models/gpt_0.pt' \
  --model_path_1 './pretrained_models/gpt_1.pt' \
  --model_path_2 './pretrained_models/gpt_2.pt' \
  --model_path_3 './pretrained_models/gpt_3.pt' \
  --init_body_pose_code 128 \
  --init_hands_pose_code 258 \
  --processed_dataset_dir './Data/SG_processed'\
  --sg_codebook './SG_pipeline/all_mocap_extracted_new.npz'\
  --retrieval_model_path <the global path to the retrieval model folder>
  ```

### Training (for Motion Tokenizer and Gesture Generator)

#### Data Preparation

Prepare the dataset for training:

1.	Download the dataset from [google drive](https://drive.google.com/file/d/1_-36bUbpOl2eC67o14EPQ5_ZhOnggA9q/view?usp=sharing), and place them in the `SG_code/Data/SG_Data/zeroeggs` directory. The directory structure should look like this:
  - SG_code
    - Model
      - ...
    - Data
      - SG_Data
        - zeroeggs
          - zeroeggs
            - ...
          - SG
            - ...

2.	Run the following command to preprocess the data:

```bash
cd SG_code

python prepare_data.py --data_dir Data/SG_Data/zeroeggs --save_dir Data/sg_processed
  ```

  The processed results will be saved in the `SG_code/Data/sg_processed` directory.

#### Training RVQ (Corresponding to SG_code/Model/residual_vq.py)
You can train the Resudial-VQVAE model by running the following command:
```bash
cd SG_code

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 main_motion_vqvae.py --config Config/motion_rvq_train.yaml --train
  ```
And then you will get a rvq model as a motion tokenizer.

####  Training GPT
After you get the rvq model, please set the `vqvae_weight` to the path of the rvq model in below files:
- `SG_code/Config/motion_gpt.yaml`
- `SG_code/Config/motion_fine_gpt_rq_level_1.yaml`
- `SG_code/Config/motion_fine_gpt_rq_level_2.yaml`
- `SG_code/Config/motion_fine_gpt_rq_level_3.yaml`

Then you need to train 4 GPT models (gesture generator) for 4 different RVQ layers by running the following command one by one:

##### Layer 1 (Corresponding to SG_code/Model/cross_cond_gpt2_2part.py):
```bash
cd SG_code

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 main_motion_vqvae.py --config Config/motion_gpt.yaml --train
  ```

##### Layer 2-4 (Corresponding to SG_code/Model/fine_gpt2_2part.py):
```bash
cd SG_code

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 main_motion_gpt.py --config Config/motion_fine_gpt_rq_level_1.yaml --train

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 main_motion_gpt.py --config Config/motion_fine_gpt_rq_level_2.yaml --train

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 main_motion_gpt.py --config Config/motion_fine_gpt_rq_level_3.yaml --train
  ```

Note: These four commands can be executed together.

  
## License
This project is licensed under the MIT License.

<!-- [![Star History Chart](https://api.star-history.com/svg?repos=heyuanYao-pku/Control-VAE&type=Date)](https://star-history.com/#heyuanYao-pku/Control-VAE&Date) -->

