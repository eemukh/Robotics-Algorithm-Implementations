% variable setup
delta_t = 0.1; 

robot_conf = [0;0]; % robot position
mu_est = [0;0];     % estimated position
sig_est = [0.005, 0; 0, 0.005]; % estimated covariance

R = [0.005, 0; 0, 0.005];
Q = [0.05, 0; 0, 0.05];

t=0;

figure(1); hold on; grid on;
title('Live Robot Position');
xlabel('X'); ylabel('Y');

% Plot empty dummy points (NaN) to set up the legend correctly (Used Gen AI
% for this legend tool)
h1 = plot(NaN, NaN, 'g.', 'MarkerSize', 10); % Dummy True
h2 = plot(NaN, NaN, 'r.', 'MarkerSize', 10); % Dummy Measured
h3 = plot(NaN, NaN, 'b.', 'MarkerSize', 10); % Dummy Estimated

xlim([0, 25]); % Locked limits moved here
ylim([0, 25]); % Locked limits moved here

% Turn AutoUpdate OFF so the loop doesn't add many keys
legend([h1, h2, h3], 'True Position', 'Measured Position', 'Estimated Mean', 'Location', 'best', 'AutoUpdate', 'off');

figure(2); hold on; grid on;
title('Estimation Error over Time');
xlabel('Time (seconds)'); ylabel('Euclidean Error')

xlim([0, 10]);
ylim([0, 0.5]);

% Plot empty dummy point for the error legend. (Used Gen AI
% for this legend tool)
h4 = plot(NaN, NaN, 'b.', 'MarkerSize', 10); 
legend(h4, 'Estimation Error', 'Location', 'best', 'AutoUpdate', 'off');

initial_theta = atan2(2, 2); 
robot_plot = DrawRobot(2.5, 5, robot_conf(1), robot_conf(2), initial_theta);

while t < 10
    input = [2;2];
    A = eye(2);
    B = [delta_t, 0; 0, delta_t];
    C = eye(2);
    
    % Transpose the noise to make them 2x1 column vectors due to mvnrnd
    % function
    epsilon = mvnrnd([0,0], R)';
    delta_lam = mvnrnd([0,0], Q)';

    robot_conf = (A * robot_conf) + (B * input) + epsilon;

    sensor = (C * robot_conf) + delta_lam;

    [mu_est, sig_est] = kalman_filter(mu_est, sig_est, input, sensor);
    
    disp('Estimated State Mean:');
    disp(mu_est);
    disp('Estimated State Covariance:');
    disp(sig_est); % Added to satisfy the print requirement
    
    % Calculate the current Euclidean error
    current_error = sqrt((robot_conf(1) - mu_est(1))^2 + (robot_conf(2) - mu_est(2))^2);
    
    % Update Figure 1 (Positions)
    figure(1);
    plot(robot_conf(1), robot_conf(2), 'g.', 'MarkerSize', 10); % True
    plot(sensor(1), sensor(2), 'r.', 'MarkerSize', 10);         % Measured
    plot(mu_est(1), mu_est(2), 'b.', 'MarkerSize', 10);         % Estimated

    delete(robot_plot);
    
    % Draw the new robot at the updated true position
    theta = atan2(input(2), input(1));
    robot_plot = DrawRobot(1, 2, robot_conf(1), robot_conf(2), theta);
    
    % Update Figure 2 (Error)
    figure(2);
    plot(t, current_error, 'b.', 'MarkerSize', 10);
    
    drawnow;    
    
    t = t + delta_t; % Increment time
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

function plot_fig = DrawRobot( width, height, center_x, center_y, theta)
    corner1 = [center_x - 0.25*height * cos(theta) + (width/2) * sin(theta), center_y - (0.25*height)*sin(theta) - (width/2)*cos(theta)];
    corner2 = [center_x - 0.25*height*cos(theta) - width/2*sin(theta),center_y - 0.25*height*sin(theta) + width/2*cos(theta)];
    corner3 = [center_x + 0.75*height*cos(theta) - width/2*sin(theta),center_y + 0.75*height*sin(theta) + width/2*cos(theta)];
    corner4 = [center_x + 0.75*height*cos(theta) + width/2*sin(theta),center_y + 0.75*height*sin(theta) - width/2*cos(theta)];
    corners = [corner1;corner2;corner3;corner4;corner1];
    corners = transpose(corners);
    x = [center_x, center_y];
    y = [center_x+height*cos(theta), center_y+height*sin(theta)];
    plot_fig(1) = plot([x(1), y(1)], [x(2), y(2)], 'k-'); hold on;  
    plot_fig(2) = plot(corners(1,:),corners(2,:),'b'); 
end
