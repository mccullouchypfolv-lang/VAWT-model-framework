# 直接拼接条件特征的对照接口

新增 `flow_model.direct_conditioning`，默认false，旧双分支检查点缺少该字段时仍按false恢复。

- false：非线性主干的升维输入为残差状态g_t与Fourier时间编码，LF/高度信息通过原FiLM条件场注入。
- true：在上述输入后额外拼接现有条件特征场：LF曲线、LF展向聚合特征、高度加权摘要、sin/cos方位角坐标。原FiLM路径全部保留，独立线性旁路不变。

默认维度由 `[B,35,N_theta]` 增至 `[B,80,N_theta]`；拼接只改变通道数，不固定方位角网格长度。输入仍只来自风速、TSR、20层弦长/厚度比及固定坐标，没有添加真实HF曲线或仿真载荷作为输入。

网络先初始化共同模块，再扩展首个升维卷积并拷贝原状态/时间输入列与偏置。因此相同随机种子下，两版本共同参数的初始值完全一致；新增45个输入通道在默认宽度下增加11520个权重。不能将false检查点通过改一个布尔值直接当作已训练的true检查点，新增版本需要训练。

## 两套独立配置

- 对照：`configs/single_blade_film_control.json`
- 直接拼接：`configs/single_blade_film_concat.json`

两套配置除开关和输出目录外完全相同：GPU、batch256、seed42、最多500轮、每轮1次更新、早停耐心90、最小改善1e-4、dopri5容差1e-5。NAGO源资产、kFFM配对、训练/验证/测试划分不变。两者共用已有第一、二阶段资产。

```console
python -u run_single_blade.py stage3 --config configs/single_blade_film_control.json
python -u run_single_blade.py stage3 --config configs/single_blade_film_concat.json
```

两条命令应依次运行，避免同时占用GPU。输出目录分别为 `stage3_film_control_dopri5` 和 `stage3_film_concat_dopri5`，均位于当前work目录下。

当前历史500轮模型的最佳权重按Heun验证选择；严格对比新接口时，应使用上述匹配的dopri5对照配置，避免同时混入求解器改变带来的选模差异。

预测时使用对应配置，检查点里的开关会自动恢复正确输入维度，普通工况预测和联合网格预测都经过同一接口。

```console
python run_single_blade.py predict --config configs/single_blade_film_concat.json --parameters configs/single_blade_query.json --n-theta 640 --output artifacts/concat_query640.npz
```

代码测试覆盖了直接路径的作用隔离、LF/高度梯度、原始检查点配置兼容、跨角度点数与检查点重建。当前默认选择直接拼接接口；仅FiLM接口保留用于对照和兼容历史检查点。当前2000轮预算配置为`configs/single_blade_film_concat_lhs_e2000.json`，仅FiLM对照为`configs/single_blade_film_only_lhs_e2000.json`；两者共享原LF基座、HF LHS划分和源资产，仅direct_conditioning及输出目录不同。实验权重和结果不纳入仓库。
