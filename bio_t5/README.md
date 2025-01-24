# BioT5 as MolNet Predictor

## 环境初始化
```bash
conda create -n biot5_pred python=3.8
# install the Pytorch compatible with cuda
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 
pip install -r requirements.txt
```

## Checkpoint路径
MolNet的checkpoints保存在 `/blob2/v-qizhipei/checkpoints/biot5/{task_name}`，其中`task_name`需要对应替换为`bace`, `bbbp`, `clintox_ct_tox`, `tox21_xxx` (xxx 表示tox21的某一个子任务，共12个).

## 示例
### 按照BioT5训练的格式重构输入
主要参考`process_pkl.py`。
这里以SciGPT的输出`test.instruct.increase_bbbp.tsv.response.pkl`文件为例，其中每个item包含了SciGPT的prompt和response，且response中包含输出的分子（可能有多个，这里以1个为例）。我们定义该任务对应的`task`为`increase_bbbp`。重构主要包含以下几步：
1. 读取要作为BioT5模型预测的分子SMILES
2. 使用RDKit进行canonicalize，然后转为SELFIES。如果生成的分子不合法，直接跳过。
3. 在开头和末尾分别添加`<bom>`和`<eom>`
4. 将3的输出嵌入到BioT5的输入模版，然后存储到`tasks/increase_bbbp`和`splits/increase_bbbp`。注意`to_json`函数的`dataset_name`参数需要进行相应变换。

### 进行Inference
主要参考`eval_molnet.sh`，并对应修改如下`ckpt_path`，`task`和`output_dir`参数即可。BioT5预测的结果会存储在`output_dir/biot5_pred.tsv`中。