# VLMaps 子模块概览与依赖关系

本文梳理 `vlmaps/` 下各模块的职责、核心数据结构、调用关系与典型数据流，便于理解与扩展。

## 目录速览

- controller/
  - controller.py：导航控制器基类 `NavController`
  - discrete_nav_controller.py：离散动作控制器（turn_left/right, move_forward）
  - continuous_nav_controller.py：连续动作控制器（速度/角速度）
- dataloader/
  - habitat_dataloader.py：Habitat 与地图坐标/姿态互转、裁剪障碍图与可视化辅助
- lseg/
  - additional_utils/, modules/：LSeg 语义特征网络与块（`LSegEncNet` 等），由建图使用
- map/
  - map.py：地图抽象基类 `Map`（路径、体素网格、障碍图、常用几何与检索 API）
  - vlmap.py：视觉-语言体素地图实现 `VLMap`（加载/索引/障碍自定义）
  - vlmap_builder.py：基于“移动底盘位姿”建 3D 体素图 `VLMapBuilder`
  - vlmap_builder_cam.py：基于“相机位姿（全局）”建 3D 体素图 `VLMapBuilderCam`
  - clip_map.py, gradcam_map.py, gtmap.py：早期/替代地图实现（以 2D 为主，当前工程核心为 `VLMap` 3D 路线）
  - interactive_map.py：交互式地图拾取与 Habitat 机器人初始状态辅助
- navigator/
  - navigator.py：基于可见图（visibility graph）的 2D 规划器 `Navigator`
- robot/
  - lang_robot.py：语言机器人接口基类，封装“面向对象/方位”的高层动作
  - habitat_lang_robot.py：Habitat 集成的语言机器人实现（仿真执行、路径跟随、可视化）
- task/：Habitat 任务封装（ObjectNav/SpatialGoal 等）
- utils/
  - mapping_utils.py：坐标/网格/投影/点云与 H5DF 读写等几何与 I/O 工具
  - clip_utils.py：CLIP 文本/图像/多模板编码与相似度
  - lseg_utils.py：像素级 LSeg 特征抽取
  - index_utils.py：类别近邻/连通域提取/动态障碍生成等
  - habitat_utils.py：Habitat 配置/状态转换/可视化
  - navigation_utils.py：可见图构建、到边界距离等
  - 其它：可视化、计时、类别表等

---

## 核心数据结构与文件

- 3D 体素地图（H5DF）
  - 路径：
    - 移动底盘位姿：`<scene>/vlmap/vlmaps.h5df`
    - 相机全局位姿：`<scene>/vlmap_cam/vlmaps_cam.h5df`
  - 数据集键：
    - `mapped_iter_list`：已融合的帧 ID 列表
    - `grid_feat`：(N, C) LSeg 语义特征
    - `grid_pos`：(N, 3) 体素格坐标（row, col, height）
    - `weight`：(N,) 特征融合权重
    - `occupied_ids`：(gs, gs, vh) -> 占据索引（-1 表示空）
    - `grid_rgb`：(N, 3) 体素颜色（可选）
    - `pcd_min/pcd_max/cs`：Camera-base 构图的包围盒与步长（cam 版本）
- Map 运行态
  - `Map`/`VLMap`：
    - `gs`（grid_size）、`cs`（cell_size）
    - `grid_feat/grid_pos/weight/occupied_ids/grid_rgb`
    - `obstacles_map/obstacles_cropped` 与裁剪边界 `rmin/rmax/cmin/cmax`
    - 坐标变换：`base2cam_tf`、`base_transform`（统一 x 前、y 左、z 上）

---

## 模块职责与关键 API

### map/

- Map（`map.py`）
  - 负责：通用地图生命周期（路径、变换）、障碍图生成/裁剪、RGB 俯视图生成、与语义目标相关的几何检索与辅助导航 API
  - 关键方法：
    - `generate_obstacle_map(h_min, h_max)`：由 3D 占据体素投影为 2D 可通行栅格（1 可行/0 占据）
    - `generate_cropped_obstacle_map()`：裁出有效区域（紧包围）
    - `generate_rgb_topdown_map()`：由 `grid_rgb` 投影彩色俯视图
    - 导航辅助：`get_pos`（由子类实现类别定位）、`get_nearest_pos`、`get_left/right_pos`、`get_pos_in_between`、`get_delta_angle_to` 等
- VLMap（`vlmap.py`）
  - 负责：加载/创建 3D VLMap、CLIP/LSeg 初始化与类目索引、动态障碍自定义
  - 关键方法：
    - `create_map(data_dir)`：按 `pose_info.pose_type` 分派至 `VLMapBuilder`/`VLMapBuilderCam`
    - `load_map(data_dir)`：读取 H5DF 到内存
    - `init_categories(categories)`：对全体体素特征与类目做一次性相似度矩阵缓存
    - `index_map(language_desc, with_init_cat)`：基于缓存或临时文本，返回 3D 体素级命中 mask
    - `customize_obstacle_map(potential_obstacle_names, obstacle_names)`：根据文本类别在 3D 上筛出动态障碍，并回投影到 2D
- VLMapBuilder（`vlmap_builder.py`，移动底盘）/ VLMapBuilderCam（`vlmap_builder_cam.py`，相机全局）
  - 负责：
    - 逐帧：深度 -> 点云（相机系），位姿变换 -> 全局，像素对齐 LSeg 特征采样，带距离衰减的特征/颜色融合
    - 体素ID分配与累积，临时/最终落盘 H5DF
  - 依赖：`lseg_utils.get_lseg_feat`、`mapping_utils.*`（投影/坐标/存取等）

### dataloader/

- VLMapsDataloaderHabitat
  - 负责：Habitat <-> 地图坐标/姿态互转；提供裁剪障碍图、彩色俯视图；在无外部 Map 时自动加载
  - 关键方法：
    - 输入：`from_habitat_tf`、`from_full_map_pose`、`from_cropped_map_pose`
    - 输出：`to_habitat_tf`、`to_full_map_pose`、`to_cropped_map_pose`

### navigator/

- Navigator
  - 负责：基于裁剪障碍图构建可见图（visibility graph），并在裁剪图上规划，再平移回全图
  - 关键方法：`build_visgraph(obstacle_map, rowmin, colmin)`、`plan_to(start_full, goal_full)`、`shift_path`

### controller/

- DiscreteNavController
  - 负责：将路径点（全图 row/col）转为离散动作序列；并可在假设下预测位姿轨迹
  - 关键方法：`convert_goal_to_actions`、`convert_paths_to_actions`、`predict_poses_with_actions`
- ContinuousNavController
  - 负责：按线速度/角速度生成连续动作参数（动作、距离/角度、时长）

### robot/

- LangRobot（接口）/ HabitatLanguageRobot（实现）
  - 负责：
    - 统一“语义-空间”导航原语：如 `move_to_object/with_object_on_left/move_in_between/face/...`
    - 在 Habitat 中执行：`setup_scene` -> `setup_map` -> `Navigator.build_visgraph` -> `move_to/turn/execute_actions`
    - 分布图（2D/3D）计算与可视化，支持“对象/区域/图像/声音”等多模态（其中 3D 基于体素集合与位置衰减）
  - 关键流程：`move_to` 内部会获取当前位姿、规划路径、经控制器生成动作、执行并记录

### lseg/

- LSegEncNet 与配套模块
  - 负责：像素级语义特征（用于 3D 体素特征融合）；首次使用会自动下载权重

### utils/

- mapping_utils：坐标/姿态/投影/点云/分辨率缩放/H5DF 存取/相机内参等
- clip_utils：文本/图像/模板多路编码，`get_lseg_score`（CLIP 文本与 LSeg 特征相似度）
- lseg_utils：`get_lseg_feat` 像素特征提取
- index_utils：`find_similar_category_id`（可调用 OpenAI 接口兜底）、`get_segment_islands_pos`、`get_dynamic_obstacles_map_3d`
- habitat_utils：Habitat 配置与状态互转、可视化
- navigation_utils：可见图构建与路径工具

---

## 模块关系与典型数据流

1) 线下建图（可选）

- 输入：RGB、Depth、位姿（base 或 camera）
- `VLMapBuilder`/`VLMapBuilderCam`：深度回投影 -> 全局点云 -> 投影采样 LSeg 特征 -> 体素融合 -> 写入 H5DF

2) 线上加载与初始化

- `VLMap.load_map()` 读取 H5DF -> `Map.generate_obstacle_map()` -> `generate_cropped_obstacle_map()`
- 若需动态障碍：`VLMap.customize_obstacle_map()` 基于类别在 3D 上筛除

3) 规划与执行（HabitatLanguageRobot）

- `VLMapsDataloaderHabitat` 负责位姿互转
- `Navigator.build_visgraph()` 基于裁剪障碍图构建可见图
- `move_to(pos_full)` -> `Navigator.plan_to(...)` -> `DiscreteNavController.convert_paths_to_actions(...)` -> `execute_actions(...)`

4) 语义检索与交互

- `VLMap.init_categories()` + `index_map()` / `get_pos()`：按语言类别获取位置/轮廓/分布
- `interactive_map.py`：从俯视图交互拾取初始位姿与目标点，辅助下游导航

---

## 坐标/索引约定（易混点）

- 网格坐标（row, col, height）：
  - row/col 以地图中心为原点，`cs` 为步长；`occupied_ids` 形状为 (gs, gs, vh)
- 基座（base）坐标：x 前、y 左、z 上；与网格互转见 `base_pos2grid_id_3d`、`grid_id2base_pos_3d`
- 角度定义：
  - 地图角度：向上为 0 度，顺时针为正（具体见 `Map.get_delta_angle_to` 等实现）
- 裁剪区域：
  - `rmin/rmax/cmin/cmax` 用于在裁剪图与全图间换算（`Navigator` 内部有 shift 与转换）

---

## 关键用法备忘

- 构图：
  - Mobile-base：`VLMapBuilder(...).create_mobile_base_map()` -> 生成 `<scene>/vlmap/vlmaps.h5df`
  - Camera-base：`VLMapBuilderCam(...).create_camera_map()` -> 生成 `<scene>/vlmap_cam/vlmaps_cam.h5df`
- 加载与查询：
  - `m = Map.create(cfg.map_config)` -> `m.load_map(scene_dir)` -> `m.generate_obstacle_map()`
  - `m.init_categories([...])` -> `m.index_map("chair")` 或 `m.get_pos("table")`
- 机器人端到端（Habitat）：
  - `robot = HabitatLanguageRobot(cfg)` -> `robot.setup_scene(scene_id)` -> `robot.move_to_object("sofa")`

---

## 扩展建议

- 新的建图来源：复用 `VLMapBuilder` 流水线（或在 `VLMapBuilderCam` 基础上用稀疏/稠密重建）
- 新的语义后端：
  - 在 `clip_utils.get_lseg_score` 注入新的文本/图像编码器
  - 或在 `VLMap` 中覆写 `init_categories/index_map`
- 自定义障碍：
  - 在 `index_utils.get_dynamic_obstacles_map_3d` 中添加新的类别筛选/聚合逻辑
- 控制/规划替换：
  - 实现新的 `NavController`；或替换 `Navigator` 以支持栅格 A*/RRT*/骨架提取等

---

## 依赖关系摘要（高层）

- 建图：lseg_utils + mapping_utils -> vlmap_builder(_cam) -> H5DF
- 地图：map(Map/VLMap) 读取 H5DF；clip_utils/index_utils 提供语义索引与动态障碍
- 数据桥：dataloader 负责姿态与裁剪换算
- 规划控制：navigator(visibility graph) + controller(discrete/continuous)
- 机器人：robot 组装 Map + Dataloader + Navigator + Controller + Habitat
- 交互：interactive_map 结合 dataloader 与 vlmap 进行拾取/可视
