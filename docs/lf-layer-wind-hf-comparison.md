# 分层来流LF与HF修正试验结果

本次已训练版本：第一阶段四标量拼接[B,4]、20层未诱导来流、sin/cos角坐标、块内FiLM、不输入高度或权重；不含后续独立四标量、块后FiLM或GridEmbeddingND实验。

LF最佳83轮，共98轮。LF测试RMSE：Cl=0.0569077、Cd=0.00395522、CM=0.000258041。该LF已由用户单独选定，记录在configs/selected_lf_model.json。

随后重新推理HF100对应工况，重建残差与训练折归一化，使用同一固定FLORAL源、独立耦合、时间加权、batch16、每轮32次更新重新训练FM。HF按原几何划分71/14/15，保持相同测试case ID、角网格和HF标签；LF预测随新LF改变。

FM最佳165轮，共255轮。15条测试样本、32次采样均值，原始系数RMSE：

| 通道 | 原完整HF默认 | 新LF对应FM | 变化 |
|---|---:|---:|---:|
| Cl | 0.229414 | 0.227826 | 降低0.69% |
| Cd | 0.0119284 | 0.0132895 | 升高11.41% |
| CM | 0.00137847 | 0.00146866 | 升高6.54% |

新模型NRMSE（除以HF真值RMS）：Cl=2.162%、Cd=4.955%、CM=2.508%。低TSR<=2的4条测试样本，Cl RMSE降低4.36%，Cd升高31.29%，CM升高10.57%。

结论：LF自身拟合更好并未自动带来HF修正整体更好。保留原完整HF默认模型（configs/selected_model.json），不得将新选定LF直接混入原FM权重。此结果是单种子固定划分的一次对照，未证明新结构的普遍优劣。

后续独立四标量+块后FiLM+角向GridEmbeddingND代码已准备，配置configs/single_blade_lf_separate_scalars_layer_wind.json；尚未训练。
