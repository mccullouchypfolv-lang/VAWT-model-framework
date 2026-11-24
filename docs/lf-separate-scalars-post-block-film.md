# 四个独立标量与块后FiLM

输入：20层未诱导来流[B,20]、方位角[B,N]、U_ref/TSR/chord/thickness_ratio四个独立[B,1]张量。不输入高度或积分权重。

每层顺序为 spectral+pointwise+local -> GELU（FNO block结束）-> 独立FiLM -> 下一block。

四个条件分别经过1->width->2width网络，生成各自gamma和beta，求和形成块后调制 (1+gamma)*h+beta；原始四个标量不拼接。每个block都有自己的四组条件网络。调制后无额外激活，下一block处理已调制场。最终沿高度平均并输出三条曲线。

配置 configs/single_blade_lf_separate_scalars_layer_wind.json；独立work。尚未训练，不改变当前默认模型或正在运行的四标量拼接版本。字段film_placement=post_block写入checkpoint模型配置。

## 自动角坐标（本次更新）

使用官方neuraloperator 1.0.2的GridEmbeddingND源码，原文件及许可证放在lf_operator/_vendor，来源URL和SHA256另存neuralop_embeddings_source.json。没有安装或变更正在训练进程的Python依赖。

仅角维使用dim=1：将20层临时折入batch后自动追加角网格，再还原为[B,2,20,N]，两个通道为分层来流与自动角坐标。不追加高度，不再拼接sin/cos。角坐标采用官方linspace(0,1,N+1)[:-1]，即角网格索引/N；原物理角仍为2*pi*(索引+0.5)/N，样本和标签不移动、不插值。拒绝非标准/移相角网格，防止自动坐标与数据错位。动态分辨率重建官方网格；本配置记录positional_embedding=grid-nd-angular-v1。

这是坐标表示变化，不能据此保证精度提升或端点周期连续。原默认模型、正在运行的前版训练均不替换。本新架构尚未训练。
