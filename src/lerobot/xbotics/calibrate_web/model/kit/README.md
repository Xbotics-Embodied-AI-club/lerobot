# 星禾套件 SO101 从臂模型

`kit_v1_so101.urdf` 与 `assets/` 里它的可见网格取自 [Xbotics-SO101-Sim](https://github.com/Xbotics-Embodied-AI-club/Xbotics-SO101-Sim)
（`so101_sim/robots/kit_assets/`，提交 `239b38e`，MIT），与仿真里的机器人同一份：官方 SO101 从臂加上套件的底板、
型材架、顶视相机支架与腕部相机支架。碰撞网格（`*_erode.obj` 等）网页上用不到，没有复制。

仿真那边改了模型，就重新复制 URDF 与它 `<visual>` 引用的网格。
