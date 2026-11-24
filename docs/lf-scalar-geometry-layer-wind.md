# 第一阶段：标量几何与分层来流

已按用户要求实现独立输入接口，已启动独立GPU训练，不替换当前默认模型。

- 条件 a=[U_ref,TSR,chord,thickness_ratio]，形状[B,4]；几何仅输入两个标量，逐层不一致时拒绝。
- 输入 x 为20层未受诱导影响的来流速度，形状[B,20]。高度仅用于数据准备时计算 U_k=U_ref*(z_k/z_ref)^alpha，不传入网络；也支持既定uniform风型。
- 现有配置：alpha=0.2，z_ref=115.6m，叶片底部59.6m，高度112m，均匀20层中心。
- 局部风速使用训练集参考风速的均值/标准差统一缩放，不能逐层单独消除风切变；标量几何使用现有训练集几何统计量。
- lifting接收[U_k,sin(theta),cos(theta)]三通道，形状[B,3,20,N_theta]；a通过FiLM进入各块。
- 不输入z或weights，沿高度直接平均；原非周期高度延拓、周期角向FNO与输出头保留。
- 输出三条曲线以及field_features和rotor_features，维度兼容下游。新特征数值不同，后续残差资产及速度场必须重新构建/训练，不能混用旧权重。
- 原始缓存仍可保留20层几何与网格信息以便验证/标签构造；新网络不接收这些重复几何或高度/权重。

## 第一阶段实验入口

配置 configs/single_blade_lf_scalar_geometry_layer_wind.json 使用独立work目录。先运行prepare，再stage1：

```console
python -u run_single_blade.py prepare --config configs/single_blade_lf_scalar_geometry_layer_wind.json
python -u run_single_blade.py stage1 --config configs/single_blade_lf_scalar_geometry_layer_wind.json
```

以上为第一阶段实验入口。第二阶段已支持在新work中根据新LF直接生成配对残差与固定FLORAL源，无需运行NAGO拟合或复制旧残差。自动训练脚本 tmp/train_compact_lf_and_flow.py 依次运行三个阶段并保存默认模型对比；已启动。

验证：不同角分辨率推理、与缓存加载输入一致、反向传播、无高度/权重接口、层间几何不一致拒绝、旧接口回归。

本次训练：LF batch256，最多100轮，patience15；FM batch16、每轮32次更新、最多2000轮、patience90；seed42，原geometry划分。只读复用原参数/标签缓存，独立重训LF和FM。日志 artifacts/single_blade_compact_lf_and_flow.log。
