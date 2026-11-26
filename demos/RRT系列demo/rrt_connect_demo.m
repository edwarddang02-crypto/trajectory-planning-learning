%% 机械臂与场景初始化
clear;
clc;
close all;
rng(1);
%机械臂初始化
robot = loadrobot("frankaEmikaPanda","DataFormat","row");  %规定统一用一行来表示一个关节配置

%% 添加工具参考坐标系（TCP） 
tcpBody = rigidBody("demo_tcp");
tcpJoint = rigidBodyJoint("demo_tcp_joint","fixed");
setFixedTransform(tcpJoint, trvec2tform([0,0,0.10]));

tcpBody.Joint = tcpJoint;
addBody(robot,tcpBody,"panda_hand");
qReference = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4]; 
fingerPosition = [0.02, 0.02];
endEffector = "demo_tcp";

         %转换为齐次变换矩阵
%% 运动学求解
% 位置，单位：m
pStart = [0.50, -0.30, 0.40];
pGoal  = [0.50,  0.30, 0.40];

% 朝向，单位：rad
% 使用ZYX欧拉角，数组顺序为 [yaw, pitch, roll]
eulStart = [0, 0, pi];
eulGoal  = [0, 0, pi];

% 构造完整位姿：平移矩阵 × 旋转矩阵
TStartDesired = trvec2tform(pStart) * ...
                eul2tform(eulStart, "ZYX");

TGoal = trvec2tform(pGoal) * ...
        eul2tform(eulGoal, "ZYX");

pObject = pGoal - [0,0,0.08]; %小球中心位置
objectRadius = 0.025;
%调用逆运动学 求两侧位置的逆解

 ik = inverseKinematics("RigidBodyTree",robot);
 weights = [1,1,1,1,1,1]; %等权重

% 给两侧位置分别设置较合适的初始猜测
qSeedStart = qReference;
qSeedStart(1) = -0.6; %对第一个关节进行初始猜测

qSeedGoal = qReference;
qSeedGoal(1) = 0.6;

[qStartFull, startInfo] = ik( ...
    endEffector, TStartDesired, weights, ...
    [qSeedStart, fingerPosition]);

[qGoalFull, solInfo] = ik( ...
    endEffector, TGoal, weights, ...
    [qSeedGoal, fingerPosition]);  %调用逆解求解器：在给定的猜测下数值求解

% 固定手指位置
qStartFull(8:9) = fingerPosition;
qGoalFull(8:9) = fingerPosition;

% 提取七维规划配置
qStart = qStartFull(1:7);
qGoal = qGoalFull(1:7);
 %正运动学检查逆解误差
 TActual = getTransform(robot,qGoalFull,endEffector);
 pActual = TActual(1:3,4).'; 
 positionError = norm(pActual - pGoal);

 fprintf("TCP目标位置误差：%.6f m\n", positionError);

%% 读取七个机械臂关节的限位
jointLimits = zeros(7, 2);

for i = 1:7
    jointLimits(i, :) = robot.Bodies{i}.Joint.PositionLimits;
end

qMin = jointLimits(:, 1)';
qMax = jointLimits(:, 2)';

%% 障碍物与环境设置
obstacle = collisionBox(0.20, 0.18, 0.60);
obstacle.Pose = trvec2tform([0.50, 0, 0.30]);

world = {obstacle};

% 沿用自碰撞跳过规则
skipPairs = cell(robot.NumBodies + 1, 2);

for i = 1:robot.NumBodies
    body = robot.Bodies{i};

    skipPairs{i, 1} = body.Parent.Name;
    skipPairs{i, 2} = body.Name;
end

skipPairs(end, :) = {'panda_link7', 'panda_hand'};

maxIterations = 5000;
stepSize = 0.20; %单次扩展最大关节空间距离
resolution = 0.03; %边检查的插值间隔

% 函数句柄：固定环境参数，只保留变化的配置输入
configValid = @(q) isConfigValid( ...
    q, robot, world, jointLimits, fingerPosition, skipPairs);

edgeValid = @(qA, qB) isEdgeValid( ...
    qA, qB, resolution, ...
    robot, world, jointLimits, fingerPosition, skipPairs);

% 检查起点与终点
assert(configValid(qStart), "起始配置无效。");
assert(configValid(qGoal), "目标配置无效。");

%% 用结构体表示两棵树的节点、父节点与位置
treeA.nodes = qStart;
treeA.parents = 0;
treeA.fromStart = true;

treeB.nodes = qGoal;
treeB.parents = 0;
treeB.fromStart = false;

%% RRT-Connect 主循环

found = false;
qPath = [];

timer = tic;

for iter = 1:maxIterations

    %随机采样一个七维配置
    qRand = qMin + rand(1, 7) .* (qMax - qMin);

    %当前树 A 朝随机配置扩展一步
    [treeA, statusA, indexA] = extendTree( ...
        treeA, qRand, stepSize, edgeValid);

    %A扩展成功后，让B尝试连接A的新节点
    if statusA ~= "TRAPPED"

        qNew = treeA.nodes(indexA, :);

        [treeB, statusB, indexB] = connectTree( ...
            treeB, qNew, stepSize, edgeValid);

        %两棵树成功连接
        if statusB == "REACHED"

            found = true;

            % 分别取得：各自根节点 -> 连接点
            pathA = tracePath(treeA, indexA);
            pathB = tracePath(treeB, indexB);

            % 确认哪条路径来自起点树
            if treeA.fromStart
                pathStart = pathA;
                pathGoal = pathB;
            else
                pathStart = pathB;
                pathGoal = pathA;
            end

            % 起点 -> 连接点 -> 目标
            % 去掉 pathGoal 末尾重复的连接点，再翻转方向
            qPath = [pathStart; flipud(pathGoal(1:end-1, :))];

            break;
        end
    end

    %交换两棵树，下一轮轮到另一棵树主动探索
    tempTree = treeA;
    treeA = treeB;
    treeB = tempTree;

end

planningTime = toc(timer);

%% 输出结果
if found
    fprintf("RRT-Connect 规划成功！\n");
    fprintf("外层迭代次数：%d\n", iter);
    fprintf("两棵树总节点数：%d\n", ...
        size(treeA.nodes, 1) + size(treeB.nodes, 1));
    fprintf("路径节点数：%d\n", size(qPath, 1));
    fprintf("规划耗时：%.3f s\n", planningTime);

    pathLength = sum(vecnorm(diff(qPath, 1, 1), 2, 2));
    fprintf("关节空间路径长度：%.3f\n", pathLength);
else
    fprintf("在 %d 次迭代内未找到路径。\n", maxIterations);
end

%% 插值演示路径
assert(found, "尚未找到路径，无法播放动画。");

qDense = qPath(1, :);

for k = 1:size(qPath, 1)-1

    qA = qPath(k, :);
    qB = qPath(k+1, :);

    distance = norm(qB - qA);

    % 每段插值的关节空间距离不超过 resolution
    numSegments = max(1, ceil(distance / resolution));

    for j = 1:numSegments
        s = j / numSegments;

        qInterp = qA + s * (qB - qA);

        qDense(end+1, :) = qInterp;
    end
end

fprintf("动画配置数：%d\n", size(qDense, 1));
%计算每一帧的TCP位置
tcpPath = zeros(size(qDense, 1), 3);

for k = 1:size(qDense, 1)

    qFull = [qDense(k, :), fingerPosition];

    T = getTransform(robot, qFull, endEffector);

    tcpPath(k, :) = T(1:3, 4).';
end

figure("Color", "w", "Name", "Panda RRT-Connect");

ax = show(robot, [qDense(1, :), fingerPosition], ...
          "Frames", "off", ...
          "PreservePlot", false);

hold(ax, "on");
for i = 1:numel(world)
    show(world{i}, "Parent", ax);
end
% 目标小球
[sx, sy, sz] = sphere(20);

surf(ax, ...
     objectRadius*sx + pObject(1), ...
     objectRadius*sy + pObject(2), ...
     objectRadius*sz + pObject(3), ...
     "FaceColor", [0.85, 0.2, 0.2], ...
     "EdgeColor", "none");

plot3(ax, pGoal(1), pGoal(2), pGoal(3), ...
      "gx", "MarkerSize", 12, "LineWidth", 2);

plot3(ax, tcpPath(:, 1), tcpPath(:, 2), tcpPath(:, 3), ...
      "b--", "LineWidth", 1.5);

tcpMarker = plot3(ax, ...
    tcpPath(1, 1), tcpPath(1, 2), tcpPath(1, 3), ...
    "bo", "MarkerFaceColor", "b", "MarkerSize", 6);

axis(ax, "equal");
xlim(ax, [-0.5, 0.9]);
ylim(ax, [-0.7, 0.7]);
zlim(ax, [0, 1.1]);

view(ax, 135, 25);
grid(ax, "on");

xlabel(ax, "X / m");
ylabel(ax, "Y / m");
zlabel(ax, "Z / m");

%逐帧播放
for k = 1:size(qDense, 1)

    qFull = [qDense(k, :), fingerPosition];

    show(robot, qFull, ...
         "Parent", ax, ...
         "Frames", "off", ...
         "PreservePlot", false);

    % 更新蓝色TCP标记的位置
    set(tcpMarker, ...
        "XData", tcpPath(k, 1), ...
        "YData", tcpPath(k, 2), ...
        "ZData", tcpPath(k, 3));

    title(ax, sprintf("RRT-Connect：第 %d / %d 帧", ...
          k, size(qDense, 1)));

    drawnow;
    pause(0.03);
end

%% 函数部分：碰撞检测，树生长，树连接，路径回溯
%单个配置检测函数
function valid = isConfigValid( ...
    qArm, robot, world, jointLimits, fingerPosition, skipPairs)
    qArm = qArm(:)';

    valid = false;

    if numel(qArm) ~= 7 || any(~isfinite(qArm))
        return;
    end

    % 读取关节限位
    qMin = jointLimits(:, 1)';
    qMax = jointLimits(:, 2)';

    % 检查关节越界
    if any(qArm < qMin | qArm > qMax)
        return;
    end
    qFull = [qArm, fingerPosition];

    % 自碰撞和环境碰撞
    collisionFlags = checkCollision( ...
        robot, qFull, world, ...
        "SkippedSelfCollisions", skipPairs, ...
        "IgnoreSelfCollision", "off", ...
        "Exhaustive", "off");
    valid = ~any(collisionFlags(:));
end

%边插值检测
function valid = isEdgeValid( ...
    qA, qB, resolution, ...
    robot, world, jointLimits, fingerPosition, skipPairs)

    valid = false;

    distance = norm(qB - qA);

    %采样
    numSegments = max(1, ceil(distance / resolution));

    % 检查插值配置
    for i = 0:numSegments

        s = i / numSegments;

        qCheck = qA + s * (qB - qA);
        if ~isConfigValid( ...
                qCheck, robot, world, jointLimits, ...
                fingerPosition, skipPairs)

            return;
        end
    end
    valid = true;

end
function [tree,status,newIndex] = extendTree(tree,qTarget,stepSize,edgeValid)
    status = "TRAPPED";
    newIndex = 0;
    %在树中找离目标最近节点
    differences = tree.nodes - qTarget;
    distances = sqrt(sum(differences.^2,2));

    [distance,nearIndex] = min(distances);
    qNear = tree.nodes(nearIndex,:);
    
    %目标配置已经在树中
    if distance == 0
        status = "REACHED";
        newIndex = nearIndex;
        return;
    end
    %向目标移动
    direction = qTarget - qNear;
    if distance <= stepSize
        % 距离够近直接到达目标
        qNew = qTarget;
        reached = true;
    else
        qNew = qNear + stepSize * direction / distance;
        reached = false;
    end
    %边检查
    if ~edgeValid(qNear,qNew)
        return;
    end
    %节点入树
    newIndex = size(tree.nodes,1) + 1;

    tree.nodes(newIndex, :) = qNew;
    tree.parents(newIndex,1) = nearIndex;

    if reached
        status = "REACHED";
    else
        status = "ADVANCED";
    end
    
end

function [tree, status, newIndex] = connectTree( ...
    tree, qTarget, stepSize, edgeValid)

    status = "ADVANCED";
    newIndex = 0;

    while status == "ADVANCED"
        [tree, status, newIndex] = extendTree( ...
            tree, qTarget, stepSize, edgeValid);
    end %当还在扩展，继续
end
function path = tracePath(tree, nodeIndex)

    path = [];
    currentIndex = nodeIndex;

    while currentIndex ~= 0

        % 将当前节点插到路径最前面
        path = [tree.nodes(currentIndex, :); path];

        % 沿父节点继续回溯
        currentIndex = tree.parents(currentIndex);

    end
end