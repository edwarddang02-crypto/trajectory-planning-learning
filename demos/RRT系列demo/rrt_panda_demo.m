clear;
clc;
close all;

% 加载 Panda，使用行向量表示关节配置
robot = loadrobot("frankaEmikaPanda", "DataFormat", "row");

% 查看连杆和关节结构
showdetails(robot);

% 获取模型的默认配置
qHome = homeConfiguration(robot);
qArm = qHome(1:7);        % RRT 使用七维节点
fingerPosition = [0.02, 0.02];
qFull = [qArm, fingerPosition]; 

%读取关节限位
jointLimits = zeros(7,2);
for i = 1:7
    jointLimits(i,:) = robot.Bodies{i}.Joint.PositionLimits;
end
qMin = jointLimits(:, 1)';
qMax = jointLimits(:, 2)';

% 初始化机械臂与障碍物
qStart = [0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4];
qStartFull = [qStart, fingerPosition];

box = collisionBox(0.40,0.30,0.30);
box.Pose = trvec2tform([0.45,0.20,0.35]);

world = {box}; %数组保存障碍物

%碰撞检查修正：跳过固定坐标系重合
skipPairs = cell(robot.NumBodies + 1, 2);

for i = 1:robot.NumBodies
    body = robot.Bodies{i};

    skipPairs{i, 1} = body.Parent.Name;
    skipPairs{i, 2} = body.Name;
end

skipPairs(end, :) = {'panda_link7', 'panda_hand'};  % 额外跳过腕部与手掌之间的固定重叠

qGoal =[1.6205, 0.2110, 1.2762, -2.2039, ...
         0.1299, 2.4665, -0.6029];

assert(isConfigValid( ...
    qStart, robot, world, jointLimits, fingerPosition, skipPairs), ...
    "起始配置无效。");

assert(isConfigValid( ...
    qGoal, robot, world, jointLimits, fingerPosition, skipPairs), ...
    "目标配置无效");

%RRT 参数
rng(1);                  % 固定随机种子
maxIterations = 5000;     % 最大迭代次数
stepSize = 0.25;          % 扩展最大关节空间距离
goalBias = 0.15;          % 向目标采样概率
goalThreshold = 0.30;
resolution = 0.05;        % 边碰撞检查分辨率

%初始化树
nodes = zeros(maxIterations + 2, 7); %七维关节对应配置空间节点
parents = zeros(maxIterations + 2, 1);

nodes(1, :) = qStart;
parents(1) = 0;

nodeCount = 1;           % 当前实际存储的节点数
goalIndex = 0;
found = false;

tic;

for iter = 1:maxIterations

    if rand < goalBias
        qRand = qGoal;
    else
        qRand = qMin + rand(1, 7) .* (qMax - qMin);
    end

    differences = nodes(1:nodeCount, :) - qRand;
    distances = sqrt(sum(differences.^2, 2));
    [~, nearIndex] = min(distances);
    qNear = nodes(nearIndex, :);
    direction = qRand - qNear;
    distanceToRand = norm(direction);

    if distanceToRand < 1e-10
        continue;
    end

    moveDistance = min(stepSize, distanceToRand);  % 不能越过采样配置

    qNew = qNear + moveDistance * direction / distanceToRand;


    %边检查
    if ~isEdgeValid( ...
            qNear, qNew, resolution, ...
            robot, world, jointLimits, fingerPosition, skipPairs)
        continue;
    end


    %节点入树
    nodeCount = nodeCount + 1;
    newIndex = nodeCount;

    nodes(newIndex, :) = qNew;
    parents(newIndex) = nearIndex;


    %判断是否到达目标
    distanceToGoal = norm(qNew - qGoal);

    % 如果新节点本身已经是目标，就直接结束
    if distanceToGoal < 1e-10
        goalIndex = newIndex;
        found = true;
        break;

    end

    % 离目标较近，尝试检查最后一段连接
    if distanceToGoal <= goalThreshold

        if isEdgeValid( ...
                qNew, qGoal, resolution, ...
                robot, world, jointLimits, fingerPosition, skipPairs)

            nodeCount = nodeCount + 1;
            goalIndex = nodeCount;

            nodes(goalIndex, :) = qGoal;
            parents(goalIndex) = newIndex;

            found = true;
            break;

        end
    end

end

planningTime = toc;

%回溯
qPath = [];

if found
    currentIndex = goalIndex;
    while currentIndex ~= 0
        qPath = [nodes(currentIndex, :); qPath];

        currentIndex = parents(currentIndex);

    end

    fprintf("规划成功！\n");
    fprintf("迭代次数：%d\n", iter);
    fprintf("树节点数：%d\n", nodeCount);
    fprintf("路径节点数：%d\n", size(qPath, 1));
    fprintf("规划耗时：%.2f s\n", planningTime);

else
    fprintf("在 %d 次迭代内未找到路径。\n", maxIterations);
end
% 插值得示意图所用采样点
if found
    qDense = qPath(1, :);
    for k = 1:size(qPath, 1)-1
        qA = qPath(k, :);
        qB = qPath(k+1, :);
        distance = norm(qB - qA);
        numSegments = max(1, ceil(distance / resolution));

        for j = 1:numSegments
            s = j / numSegments;
            qInterp = qA + s * (qB - qA);
            qDense(end+1, :) = qInterp;
        end
    end
    fprintf("动画配置数：%d\n", size(qDense, 1));
end
%作图
if found

    figure("Name", "Panda RRT 路径演示");
    % 显示第一帧
    qFull = [qDense(1, :), fingerPosition];
    ax = show(robot, qFull, ...
        "Frames", "off", ...
        "PreservePlot", false);

    hold(ax, "on");

    % 显示环境障碍物
    for k = 1:numel(world)
        show(world{k}, "Parent", ax);
    end

    % 固定视角和显示范围
    view(ax, 135, 25);
    axis(ax, "equal");
    xlim(ax, [-1.2, 1.2]);
    ylim(ax, [-1.2, 1.2]);
    zlim(ax, [-1.2, 1.5]);
    grid(ax, "on");

    xlabel(ax, "X / m");
    ylabel(ax, "Y / m");
    zlabel(ax, "Z / m");

    % 逐帧更新机器人配置
    for k = 1:size(qDense, 1)

        qFull = [qDense(k, :), fingerPosition];

        show(robot, qFull, ...
            "Parent", ax, ...
            "Frames", "off", ...
            "PreservePlot", false);

        title(ax, sprintf("Panda RRT：第 %d / %d 帧", ...
            k, size(qDense, 1)));

        drawnow;
        pause(0.03);

    end

end
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