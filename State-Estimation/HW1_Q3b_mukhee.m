clear;

%definition of initial configuration of the robot (x,y,theta)
init_pos = [0,0];
init_angle = 0;
initial_conf = [init_pos, init_angle];

robot_conf = initial_conf;
goal_position = [20, 20];

mu_est = [0;0];     % estimated position
sig_est = [0.005, 0; 0, 0.005]; % estimated covariance

R = [0.005, 0; 0, 0.005];
Q = [0.05, 0; 0, 0.05];

K_v = 1;
K_h = 0.1;
delta_t = 0.1;
k = 1;
L = 0.5;

t = 0;

% Figure 1: True Position of Robot 
figure(1); hold on; grid on;
title('True Robot Trajectory'); xlabel('X (m)'); ylabel('Y (m)');
scatter(goal_position(1), goal_position(2), 100, 'k', 'x', 'LineWidth', 2); 

% Figure 2: Measured & Estimated Positions
figure(2); hold on; grid on;
title('Sensor Measurements vs. Kalman Estimate'); xlabel('X (m)'); ylabel('Y (m)');
scatter(goal_position(1), goal_position(2), 100, 'k', 'x', 'LineWidth', 2);
h_meas = scatter(NaN, NaN, 10, 'r', 'filled');
h_est  = scatter(NaN, NaN, 10, 'b', 'filled');
legend([h_meas, h_est], 'Measured Position', 'Estimated Position', 'Location', 'best', 'AutoUpdate', 'off');

% Figure 3: Estimation Error
figure(3); hold on; grid on;
title('Estimation Error over Time'); xlabel('Time (seconds)'); ylabel('Euclidean Error');
h_err = plot(NaN, NaN, 'b.', 'MarkerSize', 10);
legend(h_err, 'Estimation Error', 'Location', 'best', 'AutoUpdate', 'off');

while norm(robot_conf(1:2) - goal_position)>0.2 %close enough to position 
    % but this is changed to 0.2 to stop the chattering at the end.

    %Parameters from the discrete time dynamics
    A = eye(2);                     
    B = [delta_t, 0; 0, delta_t];
    C = eye(2);

    %Compute epsilon and delta values
    epsilon = mvnrnd([0,0], R)';
    delta_lam = mvnrnd([0,0], Q)';
    
    %Compute error in x and y positions
    err_x = goal_position(1) - mu_est(1);
    err_y = goal_position(2) - mu_est(2);
    
    %compute velocity for x and y
    v_x = K_v * err_x;
    v_y = K_v * err_y;
    
    %round out velcoty to set a boundary
    v_com_x = max(min(v_x, 2),-2);
    v_com_y = max(min(v_y, 2),-2);

    %create an input for kalman filter using computed velocities
    v_input = [v_com_x; v_com_y];
    
    %update robot configuration
    robot_conf(1:2) = ((A * robot_conf(1:2)') + (B * v_input) + epsilon)';

    % Compute sensor model with the configuration
    sensor = (C * robot_conf(1:2)') + delta_lam;

    % Compute KF algorithm
    [mu_est, sig_est] = kalman_filter(mu_est, sig_est, v_input, sensor);
    
    %Update desired angle with position information
    theta_des = atan2(goal_position(2)-robot_conf(2), goal_position(1)-robot_conf(1));
    
    if norm(v_input) > 0.05  %implementing this to stop chattering at end
        robot_conf(3) = wrapToPi(atan2(v_com_y, v_com_x));
    end

    % Update Figure 1
    figure(1); 
    if exist('robot_plot', 'var') && all(isgraphics(robot_plot))
        delete(robot_plot);
    end
    robot_plot = DrawRobot(2.5, 5, robot_conf(1), robot_conf(2), robot_conf(3)); 
    scatter(robot_conf(1), robot_conf(2), 10, 'green', 'filled'); 
    
    % Update Figure 2
    figure(2);
    scatter(sensor(1), sensor(2), 10, 'r', 'filled');         % Measured (Red)
    scatter(mu_est(1), mu_est(2), 10, 'b', 'filled');         % Estimated (Blue)
    
    % Update Figure 3 (Error)
    figure(3);
    current_error = norm(robot_conf(1:2)' - mu_est);
    plot(t, current_error, 'b.', 'MarkerSize', 10);
    
    drawnow;
    
    t = t + delta_t; % Increment time
    k = k + 1;
    
end

function plot_fig = DrawRobot( width, height, center_x, center_y, theta)
        
    corner1 = [center_x - 0.25*height * cos(theta) + (width/2) * sin(theta), center_y - (0.25*height)*sin(theta) - (width/2)*cos(theta)];
    corner2 = [center_x - 0.25*height*cos(theta) - width/2*sin(theta),center_y - 0.25*height*sin(theta) + width/2*cos(theta)];
    corner3 = [center_x + 0.75*height*cos(theta) - width/2*sin(theta),center_y + 0.75*height*sin(theta) + width/2*cos(theta)];
    corner4 = [center_x + 0.75*height*cos(theta) + width/2*sin(theta),center_y + 0.75*height*sin(theta) - width/2*cos(theta)];

    corners = [corner1;corner2;corner3;corner4;corner1];
    corners = transpose(corners);
    x = [center_x, center_y];
    y = [center_x+height*cos(theta), center_y+height*sin(theta)];
    plot_fig(1) = plot([x(1), y(1)], [x(2), y(2)], 'k-'); hold on;  % 'b-' for blue solid line

    plot_fig(2) = plot(corners(1,:),corners(2,:),'b'); xlim([-5,30]); ylim([-5,30]); xlabel('x (m)'); ylabel('y (m)');
        
end

function [mu_curr, sig_curr] = kalman_filter(mu_prev, sig_prev, input, sensor)
    delta_t = 0.1;
    R = [0.005, 0; 0, 0.005];
    Q = [0.05, 0; 0, 0.05];
    A = eye(2);
    B = [delta_t, 0; 0, delta_t];
    C = eye(2);
    
    mu_next = (A*mu_prev) + (B*input);
    sig_next = (A*sig_prev*transpose(A)) + R;
    
    kalman_gain = sig_next * transpose(C) * inv((C*sig_next*transpose(C))+Q);
    mu_curr = mu_next + kalman_gain*(sensor - (C*mu_next));
    sig_curr = (eye(2) - (kalman_gain*C)) * sig_next;
end