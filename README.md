# GUME: Graphs and User Modalities Enhancement for Long-Tail Multimodal Recommendation

<!-- PROJECT LOGO -->

## Introduction

This is the Pytorch implementation for our GUME paper:

>GUME: Graphs and User Modalities Enhancement for Long-Tail Multimodal Recommendation

## Environment Requirement
- Python 3.10 or later
- Install dependencies with `pip install -r requirements.txt`

## Dataset

We provide four processed datasets: Baby, Sports, Clothing, Electronics.

Download from Google Drive: [Baby/Sports/Clothing/Electronics](https://drive.google.com/drive/folders/1_j7du9KX30S9PwX8jmHlTmhxOof5WTnS?role=writer)

## Training
  ```
  python src/main.py
  ```

### TensorBoard

Each training run writes TensorBoard events under `runs/<model>-<dataset>/`.
For example, to train GUME on the Clothing dataset:

```bash
python src/main.py -m gume -d clothing
```

Start TensorBoard on the training server:

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

On your local computer, forward the port over SSH (replace `<username>` with your server login):

```bash
ssh -N -L 6006:127.0.0.1:6006 -p <ssh-port> <username>@<server-address>
```

Then open `http://localhost:6006`. Select multiple runs in the Scalars dashboard to compare their curves.

## Performance Comparison
<img src="image/result.png" width="900px" height="380px"/>

## Citing GUME
If you find GUME useful in your research, please consider citing our [paper](https://arxiv.org/abs/2407.12338).
```
@article{lin2024gumegraphsusermodalities,
      title={GUME: Graphs and User Modalities Enhancement for Long-Tail Multimodal Recommendation}, 
      author={Guojiao Lin and Zhen Meng and Dongjie Wang and Qingqing Long and Yuanchun Zhou and Meng Xiao},
      year={2024},
      eprint={2407.12338},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.12338}, 
}
```
The code is released for academic research use only. For commercial use, please contact [Guojiao Lin](guojiaolin37@gmail.com).

## Acknowledgement
The structure of this code is  based on [MMRec](https://github.com/enoche/MMRec). Thank for their work.
