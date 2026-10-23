# G0 裁决（机制假说 M1）

**结论：PENDING**

未提供 --assign-stats，无法检验 M1。请先用 ADEA_DIAG=1 跑一次 baseline 训练（或先跑数据集级统计确认粘连/长尾事实）。

## 数据集级事实

- 图片总数：25378（split: ['train', 'val', 'test']）
- 存在非忽略粘连（β>0.05）的图：5863（23.1%）
- 高密度图（>20 实例）：2135（8.41%）
- 平均实例/图：7.582，最大 214
- GT 对总数：3254263，其中 β>0.05 占 0.91%
- 稀有类（<500 实例）：['Nematode trench', 'Land tiger', 'eight-character tiger', 'holotrichia oblita']
- 极小类（相对尺度 <0.05%）：['Plutella xylostella', 'Rice planthopper']
