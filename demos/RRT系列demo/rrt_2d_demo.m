clear;
clc;
close all;

rng(1);
%环境、起终点
bounds = [0,10;
          0,10];
obstacles = [4,6,3,7];
qStart = [1,1];
qGoal = [9,9];
%绘图初始化
figure;
hold on;
axis equal;
xlim(bounds(1,:));
ylim(bounds(2,:));
grid on;
xlabel('x');
ylabel('y');
title('二维RRT示意');

for i = 1:size(obstacles, 1)          

    obs = obstacles(i, :);          

    rectangle('Position', ...       
        [obs(1), obs(3), ...
         obs(2)-obs(1), obs(4)-obs(3)], ...
        'FaceColor', [0.5, 0.5, 0.5]);

end
plot(qStart(1), qStart(2), ...        
    'go', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
plot(qGoal(1), qGoal(2), ...          
    'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
%RRT参数定义
stepSize = 0.3;
checkStep = 0.05;
goalBias = 0.1;
goalTolerance = 0.3;
maxIterations = 5000;
%记录节点与父节点
nodes = qStart;
parents = 0;

found = false;
goalIndex = [];

for iter = 1:maxIterations  %采样循环
    if rand < goalBias
        qRand = qGoal; %小概率以终点为采样点
    else
        qRand = bounds(:,1)' + rand(1,2) .* (bounds(:,2)-bounds(:,1))';
    end
    distances = sqrt(sum((nodes-qRand).^2,2));

    [~,nearIndex] = min(distances);
    qNear = nodes(nearIndex,:);  %找出最近节点坐标
    
    direction = qRand -qNear;
    distance = norm(direction);
    if distance < 1e-12   % 两点几乎重合
        continue;                          
    end
    
    moveDistance = min(stepSize,distance);

    qNew = qNear + moveDistance * direction/distance;

if ~isMotionValid(qNear,qNew,bounds,obstacles,checkStep)
    continue;
end   %检查新边是否碰撞

nodes(end+1,:) = qNew;
newIndex = size(nodes,1);
parents(newIndex,1) = nearIndex;

plot([qNear(1), qNew(1)], ...         
         [qNear(2), qNew(2)], ...              
         '-', 'Color', [0.65, 0.8, 1]);      
    drawnow limitrate;      % 更新图像

    distanceToGoal = norm(qNew-qGoal);
    if distanceToGoal < 1e-12                 % 新节点本身就是终点

        goalIndex = newIndex;                
        found = true;                       
        break;
    elseif distanceToGoal <= goalTolerance
        if isMotionValid(qNew, qGoal, bounds, obstacles, checkStep)
            nodes(end+1,:)=qGoal;
            goalIndex = size(nodes,1);
            parents(goalIndex,1) = newIndex;
            found = true;
            break;
        end
    end
end

%回溯
if found 
    path = [];
    currentIndex = goalIndex;
    while currentIndex ~= 0
        path = [nodes(currentIndex,:);path];

        currentIndex = parents(currentIndex); %更新当前节点为父节点
    end
plot(path(:, 1), path(:, 2), ...           % 取出路径所有点的 x 和 y
        'k-', 'LineWidth', 2.5);

    fprintf('采样次数：%d\n', iter);
    fprintf('树节点数：%d\n', size(nodes, 1));
    fprintf('路径节点数：%d\n', size(path, 1));

else                                          
    fprintf('本次未找到路径。\n');
end

function valid = isStateValid(q,bounds,obstacles)
    valid = true;
    if q(1) < bounds(1, 1) || ...
       q(1) > bounds(1, 2) || ...
       q(2) < bounds(2, 1) || ...
       q(2) > bounds(2, 2)                 

        valid = false;                    % 点在地图外
        return;

    end

    for i = 1:size(obstacles,1)

        obs = obstacles(1,:);

        insideX = q(1) >= obs(1) && q(1) <= obs(2); 

        insideY = q(2) >= obs(3) && q(2) <= obs(4);
        if insideX && insideY 
            valid = false;
            return;
        end
    end
end

function valid = isMotionValid(qA,qB,bounds,obstacles,checkStep)
    valid = true;
    distance = norm(qB-qA);
    nSegments = max(1,ceil(distance/checkStep));

    for k = 0:nSegments                   % 逐点检查碰撞

        s = k/nSegments;

        q = (1-s)*qA + s*qB;

        if ~isStateValid(q, bounds, obstacles)

            valid = false; 
            return;

        end
    end
end
