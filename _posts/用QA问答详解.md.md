# 用QA问答详解

---

---

layout: post
title: "用QA问答详解Mapping建图"
date: 2024-11-06
tags: [Perception]
comments: true
author: zihaosoog

---

## Q：什么是SDMap,HDMap以及OnlineMapping，且3者的区别是什么

1. SDMap 是指低精度地图，一般指拓扑地图，只有道路级别精度，大约在米级别，不包含路面标志和坐标，主要用于导航。
2. HDMap 是指高精度地图，侧重车道级别信息，如车道线及路面箭头等，精度在分米设置厘米级别，提供了精确定位和路径规划。
3. OnlineMapping(OM)任务属于轻地图重感知潮流的一个体现，OM任务通过感知完成在线重建高精建图从而减少对先验地图的依赖。
4. 地图数据表示形式：一般有2种，grid-based和vector-based，grid-based的地图元素常以像素点表示地图元素，类似于图片heatmap形式；vector-based的地图元素以多段线(polyline)表示，每条polyline有对应class比如roadedge，人行横道等。通常对polyline进行采样，并对采样点坐标和class编码后使用。

## Q：SDMap和HDMap如何辅助OM任务

SDMap和HDMap可以作为OM任务的先验知识，加快训练速度提高模型性能，在【P-MapNet】中，多模态多视角的BEV feat与SDMap 的polyline采样点编码作 cross-attention(transformer query)，从而实现与BEV feat的fusion，然后生成分割图作为下游query，最后在此query的基础上利用自监督的重建masked HDMap，不断refine输出到分割HDMap。

## Q：端到端实时在线矢量化高精地图构建 [MapTR]

整体框架采用encoder+decoder的DETR范式。

encoder采用LSS范式，基于多视角2D feat生成BEV feat。

decoder部分采用TransFormer+query范式，query由2部分组成，包括instance query(i)和point query(j)，两者排列组合相加得到第i个地图元素的第j个点的分层query表示为query(ij)。首先对于这些qij作MHSA，作instance之间和之内作交互，然后计算qij的2D参考点BEV坐标，基于这些offset从BEV feat中提取相应的特征，作cross-attention来update qij。

head预测分为两个分支，回归point的坐标(x,y)和分类instance的类别score，GT采用对车道线等instance进行均匀点采样，得到gt集合(class,point_set,排列组合)。其中排列组合分为2种，polyline(不闭环多段线)和polygon(闭环多段线)，基于不同起止点和顺逆时针方向 构建不同折多段线组合。

相应的二分图匹配分为2层，instance级别的匹配和point级别的匹配，instance匹配代价函数包括class匹配代价(focal)和point_xy位置匹配代价。上述匹配后instance已经被分配给相应的gt，在被分配正标签的instance中进行point级别匹配，即计算该instance中点集与哪一个排列组合的曼哈顿距离最小，至此完成全部匹配。

## Q：中融合HDmap 矢量化 OnlineMapping [VectorMapNet]

输入图像和激光，图像分支采用IPM将2D feat转换为BEV feat，为了缓解地面不是平面的假设，将特征转换到-1到2m的4个BEV 平面，最后将这些feat 相cat得到图像的图像的BEV feat。激光分支采用PointPillars的变体提取BEV feat。中融合即将上述图像和激光的BEV feat相cat，然后两层卷积得到最终的BEV feat。

采用DETR范式，query由i个地图元素query组成，每个地图元素query包括k个关键点的point query(由position embedding和class embedding组成)。将所有point query同时经过self-attention和cross-attention，预测每个关键点的坐标，然后将单个地图元素的point query进行cat，然后再MLP得到类别标签。

基于预测结果生成多段线矢量地图，生成器估计多段线每个顶点位置及起始点顺序的联合分布概率完成，具体来说采用自回归网络transformer，输入预测坐标和类别的词元化embedding，类似于NLP中预测单词的顺序和类别。