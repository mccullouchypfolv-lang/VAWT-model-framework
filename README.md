# 单叶片气动曲线：低保真基座与残差流匹配

物理输入：来流参考速度、TSR、20层弦长和厚度比。高度与方位角表示坐标。输出：单叶片Cl、Cd以及绕风轮轴的力致转矩系数CM。

本仓库只保存代码、配置与测试，不保存数据集、训练权重、缓存、训练日志、结果图或实验输出。

## 新旧速度场

- 新模型：`lf_operator/residual_flow.py`。独立线性旁路 + 非线性FiLM-FNO，4层、64通道，Fourier时间编码，条件场调制，SiLU→FiLM→GroupNorm。
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

已有第一、二阶段资产时可直接运行stage3。正式配置为GPU、batch_size=256、seed=42；旧stage3结果目录不覆盖，新速度场写入`stage3_filmfno_v2`。没有断点续训接口，已有输出目录拒绝覆盖。

## 参数与网格预测

```console
python run_single_blade.py predict-lf --parameters configs/single_blade_query.json --n-theta 320 --output artifacts/query_lf.npz
python run_single_blade.py predict --parameters configs/single_blade_query.json --n-theta 640 --output artifacts/query_hf.npz
python run_single_blade.py predict-grid --parameters configs/single_blade_grid64.json --output artifacts/grid64_hf.npz
```

联合网格输出按风速×TSR×方位角×通道排列；示例形状为`[64,64,64,3]`。`predict-grid-lf`仅计算LF基座。`--query-batch-size`只控制网格推理的工况分块，不改变训练batch。联合网格入口已编写，尚未完成运行验证。

源与残差均值在新方位角网格上按原有Fourier系数求值，保留训练已解析的源带宽；不修改训练标签或对最终预测曲线进行插值放大。模型仍固定20个高度层。预测新工况、加密参数坐标和提高方位角分辨率均需与对应真值评估精度；生成离散度尚未校准。
