# VAWT 官方 FLORAL 基准

后续模块改进以官方 `modes=64` 为基准。已完成的对照运行位于 `results/hf100_seed42/`，种子 42，共训练 243 轮，最佳权重为第 153 轮的 `best.ckpt`。在同一批 15 条 HF 测试曲线上，Cl、Cd、CM 的 RMSE 分别为 0.278625、0.026969、0.001796。模态数 6 和 80 的两次实验各自保存在独立结果目录。

在本目录的 `upstream/` 使用官方仓库提交 `9d1c1c0c25a7b367ad24441358660e98b274aa2c`，`run.py` 直接调用该版本的 `Flow` 训练和积分。

数据接入读取第 83 轮选定 LF 权重对应的已验证缓存。原 HF 按几何划分，训练、验证、测试分别为 71、14、15 条曲线。物理条件为参考风速、TSR、弦长和厚度比，在各方位角展开。训练目标为 HF−LF，按官方一维例程的通道标准化方式处理，统计量仅取训练集。官方 FiLMFNO 从条件张量和时间编码中接收信息，LF 曲线用于构造残差和还原 HF 预测。

使用官方 `oneDCorr` 配置和超参数，将 `floral` 设为 true，启用残差模式。训练使用 batch 4、最多 300 轮，按验证集 FM 损失选最佳权重。评估调用官方 `integrate_flow`，每工况 32 次采样，报告 HF 曲线 RMSE 与按 HF 真值 RMS 归一化的 NRMSE。对照组是使用同一 LF 权重的本地 FiLM-FNO，脚本逐项核验双方的案例 ID、角网格、HF 标签和 LF 基线。该实验比较两套完整实现。

首次设置时，在项目根目录用已安装 CUDA PyTorch 的 Python 执行：

```powershell
git clone https://github.com/sahilbhola14/floral.git benchmarks/floral_official/upstream
git -C benchmarks/floral_official/upstream checkout --detach 9d1c1c0c25a7b367ad24441358660e98b274aa2c
python -m venv --system-site-packages benchmarks/floral_official/.venv
benchmarks/floral_official/.venv/Scripts/python.exe -m pip install -r benchmarks/floral_official/requirements.txt
```

启动基准：

```powershell
benchmarks/floral_official/.venv/Scripts/python.exe benchmarks/floral_official/run.py --modes 64 --output benchmarks/floral_official/results/new_run
```

结果目录包含训练日志、最佳 checkpoint、验证和测试预测及 `metrics.json`。本目录的虚拟环境继承原 CUDA PyTorch 安装，附加依赖安装在本地虚拟环境。
