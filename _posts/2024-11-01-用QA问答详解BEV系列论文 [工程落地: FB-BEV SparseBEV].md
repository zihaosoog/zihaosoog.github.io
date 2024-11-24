---
layout: post
title: "用QA问答详解BEV系列论文 [工程落地: FB-BEV / SparseBEV]"
date:   2024-11-01
tags: [Perception]
comments: true
author: zihaosoog
---

## Q：前向投影和反向投影概念以及work有哪些？

前向投影：代表作LSS，思路：2D为主动，结合depth信息，投影到3D空间，生成稀疏的3D feat

反向投影：代表作BEVFormer，思路：3D为主动过，从3D query(point)投影到2D feat，生成密集但缺少深度信息depth，存在多个3D point投影到同一个2D point到匹配问题

## Q：如何利用 密集预测 提高 LSS稀疏BEV性能 实现加速工程落地 [FB-BEV]？

多视角图片作为输入，通过backbone和FPN提取2D feat，将多视角feat输入深度估计网络获得深度信息depth，基于2D feat和depth完成LSS得到BEVfeat。

基于上述LSS生成的稀疏BEV feat，进行2分类mask预测(前景/背景)，通过阈值thre选取前景区域，选取前景query(BEV grid)，在后续的BEVFormer中着重细化。

基于上述前背景的query(类似于DETR中采用topk等方法优化query)，以及估计得到的depth信息，进行BEVFormer完成密集BEV feat的预测(相比于原版BEVFormer的BEV query grid更稀疏)。具体来说，由于3D query投影时，同一投影射线上不同深度的3D点会投影到同一个2D点上，此时将3D点到车原点的距离(即深度depth)离散为表示深度depth的向量w1，而LSS中使用的depth向量表示为w2，将w1和w2相乘作为Former中attention的加权系数，实现深度约束的BEVFormer。

最终，将BEVFormer密集refine之后的前景query，即经过密集attention得到的BEV feat’，与LSS得到的稀疏的BEV feat相加，由此实现对LSS中BEV feat的进一步refine。

将前向投影和后向投影相结合，通过稀疏的前向投影BEV feat筛选前景query，feed到BEVFormer中进行优化，降低了BEVFormer因密集grid带来的计算复杂度，最终实现了提升LSS稀疏BEV的目的。

## Q：如何用 Query和Former的密集预测框架 实现稀疏BEV方案[SparseBEV]？

6视角图像作为输入，通过Backbone和FPN提取多尺度多视角2D img feat，整体架构类似于BEVFormer，初始化point query(不仅包含3D坐标，还有长宽高，旋转角度和速度)用于后续回归以及content query用于后续分类。

首先将所有query输入尺寸自适应self-attention，

$$
Atten(QKV) = softmax(\frac{QK^T}{\sqrt{d}}-\tau D)V
$$

与常规自注意力区别在于：通过做差控制注意力的范围，其中tau用来控制query的感受野，由query feat经过FC获得，D用来表示两个query之间的欧氏距离。

接下来是cross-attention，用来获取时序信息，对于query feat进行线性变换得到3D point offset。将3D point投影到2D并提取相应点的2D img feat，与此同时，将3D point wrap到其他时刻的2D帧上，采样时序的2D feat。将时序feat进行stack，然后用channel mixing和point mixing进行融合，用作下一步的回归和分类预测。

对于时序warp操作，目前分为两种，warp时序feat到t时刻和warp当前t时刻point到其他时刻帧。
