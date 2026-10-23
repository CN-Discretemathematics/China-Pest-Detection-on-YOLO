# G0 裁决（机制假说 M1）

**结论：PASS**

M1 与 M2 同时成立：面积标定后粘连目标残差 -0.7614 < 孤立目标 0.0（corr(residual, β)=-0.2472），且 corr(log 相对尺度, n_pos)=0.8521。（标定模型 E[n_pos|a]=min(11.0, 9475.99·a_rel)）→ 实施**面积标定的公平正样本分配**（SLA）：按期望标定残差，对被低估的 GT 从超额 GT 回收 anchor，并在更细的尺度层（P2/stride-4）为微目标提供足够候选。

## 数据集级事实

- 图片总数：25378（split: ['train', 'val', 'test']）
- 存在非忽略粘连（β>0.05）的图：5863（23.1%）
- 高密度图（>20 实例）：2135（8.41%）
- 平均实例/图：7.582，最大 214
- GT 对总数：3254263，其中 β>0.05 占 0.91%
- 稀有类（<500 实例）：['Nematode trench', 'Land tiger', 'eight-character tiger', 'holotrichia oblita']
- 极小类（相对尺度 <0.05%）：['Plutella xylostella', 'Rice planthopper']

## 训练期采集（n_pos vs β）

- 批次数：9524，GT 样本数：356478
- corr(n_pos, β) Spearman：0.5214 / Pearson：0.3476
- corr(n_pos, area)：0.7618
- 粘连组平均正样本：7.6145；孤立组：1.246
- 被饿死（n_pos<3）样本：290269（其中粘连目标 672，占 0.23%）
