# 单叶片气动曲线：低保真基座与残差流匹配

物理输入：来流参考速度、TSR、20层弦长和厚度比。高度与方位角表示坐标。输出：单叶片Cl、Cd以及绕风轮轴的力致转矩系数CM。

本仓库只保存代码、配置与测试，不保存数据集、训练权重、缓存、训练日志、结果图或实验输出。

## 新旧速度场

- 新模型：`lf_operator/residual_flow.py`。独立线性旁路 + 非线性FiLM-FNO，4层、64通道，Fourier时间编码，条件场调制，SiLU→FiLM→GroupNorm。
- 默认采用直接拼接：`flow_model.direct_conditioning=true`在保留FiLM条件场的同时，将LF与高度特征直接送入非线性主干。两套接口及用法见[直接拼接接口](docs/direct-conditioning.md)。
- 旧模型：`lf_operator/legacy_residual_flow.py`。单主干FiLM-FNO，默认3层、32通道，全局工况与时间调制。保留与旧检查点的参数命名兼容；当前统一训练入口使用新模型。
- LF基座：`lf_operator/model.py`，2D-to-1D FiLM-FNO。
- NAGO相关源拟合：`lf_operator/spectral_source.py`。
- kFFM相关全batch Kernel OT：`lf_operator/kernel_ot.py`。

旧权重在本地可用以下类加载，不需要把权重提交到Git：

```python
from lf_operator.legacy_residual_flow import ConditionalVelocity
model = ConditionalVelocity(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model'])
```

## 训练

先准备本地数据路径：编辑`configs/single_blade.json`。安装`requirements.txt`依赖后，从仓库根目录依次运行：

```console
python -u run_single_blade.py prepare
python -u run_single_blade.py stage1
python -u run_single_blade.py stage2
python -u run_single_blade.py stage3
```

已有第一、二阶段资产时可直接运行stage3。当前选定模型记录在`configs/selected_model.json`：FiLM＋直接拼接、modes=12，实际605轮、最优515轮。默认work为`artifacts/single_blade_parameters_v1`，flow_output为`stage3_film_concat_dopri5_e2000`，LF和HF均使用几何分组划分（HF32/9/8）。默认GPU、batch256、seed42、固定lr=1e-3、AdamW权重衰减1e-3、最大2000轮、早停90。已有输出目录拒绝覆盖，新实验需另设flow_output；没有断点续训入口。

ODE求解器由配置中的`ode_method`指定，当前为`dopri5`，`ode_atol=ode_rtol=1e-5`。`ode_steps`只控制固定步长方法，dopri5自行选择步长。容差作用于标准化残差。ODE设置用于验证与预测，不改变流匹配速度回归损失。可用`python scripts/evaluate_flow_solver.py --method dopri5`对已有模型重新评估；它固定检查点的随机种子与采样数，写入独立子目录，不更新权重。

学习率调度对照使用`configs/single_blade_film_concat_e2000_explr099.json`：

```console
python -u run_single_blade.py stage3 --config configs/single_blade_film_concat_e2000_explr099.json
```

该配置保留直接拼接、modes=12和最大2000轮，使用AdamW，`flow_weight_decay=1e-4`、`flow_lr_gamma=0.99`。ExponentialLR在每轮全部更新结束后调用一次；第e轮实际学习率为`flow_lr * gamma**(e-1)`。日志记录本轮和下一轮学习率，最优检查点同时保存优化器与调度器状态；当前入口仍不支持断点续训。历史配置若未指定这两个字段，保持原来的固定学习率和`weight_decay=0.001`。新结果使用独立目录。

HF的LHS划分对照配置为`configs/single_blade_film_concat_lhs_e2000.json`，不作为当前选定模型。该实验复用既有LF基座和49份参数缓存，在独立work目录运行`stage2`再运行`stage3`，不改变LF训练划分。`hf_split.method=latin_hypercube`在风速、TSR、弦长和厚度比上为各子集生成独立LHS目标，再用全局唯一匹配分配已有样本；这是离散数据的近似LHS覆盖，不保证观测样本精确满足分层。当前49条划分35/7/7；允许集合间几何重叠，保持样本互斥，评估范围与几何留出实验不同。训练折归一化和源拟合重新计算，NAGO/kFFM算法不变；`stage2/split_design.json`保存设计和匹配结果。代码在未指定hf_split时仍按geometry处理，以兼容历史配置。

第一阶段也采用LHS的对照配置为`configs/single_blade_all_lhs_e2000.json`：在独立work按`stage1`、`stage2`、`stage3`顺序运行。`lf_split.method=latin_hypercube`使用9580条LF的四维输入划分，不按几何分组；大型集合通过稀疏候选图最小代价唯一匹配分配样本，设计和实际匹配求解信息保存在`stage1/split_design.json`。原始标签/缓存不变，第一阶段重新初始化训练，后续LF基线和隐特征重新推理，源与残差缩放重新拟合。HF仍使用同一35/7/7划分，可在同一HF测试集上比较改变LF基座的影响。非几何划分允许相同几何跨集合，不能据此声称未见几何泛化。

## 参数与网格预测

```console
python run_single_blade.py predict-lf --parameters configs/single_blade_query.json --n-theta 320 --output artifacts/query_lf.npz
python run_single_blade.py predict --parameters configs/single_blade_query.json --n-theta 640 --output artifacts/query_hf.npz
python run_single_blade.py predict-grid --parameters configs/single_blade_grid64.json --output artifacts/grid64_hf.npz
```

联合网格输出按风速×TSR×方位角×通道排列；示例形状为`[64,64,64,3]`。`predict-grid-lf`仅计算LF基座。`--query-batch-size`只控制网格推理的工况分块，不改变训练batch。联合网格入口已编写，尚未完成运行验证。

源与残差均值在新方位角网格上按原有Fourier系数求值，保留训练已解析的源带宽；不修改训练标签或对最终预测曲线进行插值放大。模型仍固定20个高度层。预测新工况、加密参数坐标和提高方位角分辨率均需与对应真值评估精度；生成离散度尚未校准。
