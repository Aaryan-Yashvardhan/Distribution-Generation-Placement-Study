nbus=33;
%index, starting bus, ending bus, R, X
LD=[1    1      2     0.0922    0.0470
    2    2      3     0.4930    0.2511
    3    3      4     0.3660    0.1864
    4    4      5     0.3811    0.1941
    5    5      6     0.8190    0.7070
    6    6      7     0.1872    0.6188
    7    7      8     0.7114    0.2351
    8    8      9     1.0300    0.7400
    9    9      10    1.0440    0.7400
    10    10     11    0.1966    0.0650
    11    11     12    0.3744    0.1238
    12    12     13    1.4680    1.1550
    13    13     14    0.5416    0.7129
    14    14     15    0.5910    0.5260
    15    15     16    0.7463    0.5450
    16    16     17    1.2890    1.7210
    17    17     18    0.7320    0.5740
    18     2     19    0.1640    0.1565
    19    19     20    1.5042    1.3554
    20    20     21    0.4095    0.4784
    21    21     22    0.7089    0.9373
    22     3     23    0.4512    0.3083
    23    23     24    0.8980    0.7091
    24    24     25    0.8960    0.7011
    25     6     26    0.2030    0.1034
    26    26     27    0.2842    0.1447
    27    27     28    1.0590    0.9337
    28    28     29    0.8042    0.7006
    29    29     30    0.5075    0.2585
    30    30     31    0.9744    0.9630
    31    31     32    0.3105    0.3619
    32    32     33    0.3410    0.5302 ];

%bus no, Pload, Qload

BD=[1	0         0  	  0
    2	100       60 	  0
    3	90        40      0  	
    4	120       80	  0  
    5	60        30	  0  
    6	60        20      0 
    7	200       100	  0
    8	200       100	  0 
    9	60        20      0	
    10	60        20 	  0
    11	45        30 	  0
    12	60        35      0 
    13	60        35      0 
    14	120       80      0 
    15	60        10      0 
    16	60        20      0 
    17	60        20      0 
    18	90        40      0 
    19	90        40      0 
    20	90        40      0 
    21	90        40      0
    22	90        40      0 
    23	90        50      0
    24	420       200	  0
    25	420       200	  0
    26	60        25      0  
    27	60        25      0 
    28	60        20      0
    29	120       70      0
    30	200       600	  0
    31	150       70      0  
    32	210       100	  0
    33	60        40      0];
Sbase=100;                  % BASE MVA VALUE
Vbase=12.66;                   % BASE KV VALUE
Zbase=(Vbase^2)/Sbase;      % BASE IMPEDANCE VALUE
LD(:,4:5)=LD(:,4:5)/Zbase;  % PER UNIT VALUES OF LINE R AND X
BD(:,2:3)=BD(:,2:3)/(1000*Sbase);  % PER UNIT VALUES LOADS P AND Q AT EACH BUS
N=max(max(LD(:,2:3)));

V=ones(size(BD,1),1);                    % INITIAL VALUE OF VOLTAGES
Z=complex(LD(:,4),LD(:,5));     % LINE IMPEDANCE IN PER UNIT VALUES
Iline=zeros(size(LD,1),1);      % LINE CURRENT MATRIX



[Tl, TQl, Voltage, Vangle, Ploss, Qloss] = calculate_power_losses(LD,BD,  Z, V, Sbase, 200);
% Assuming Ploss and Qloss are vectors of the same length
complexVector = sum(Ploss) + 1i*sum(Qloss); % Combine real and imaginary parts
initial_loss = norm(complexVector); % Calculate the magnitude
plot(Voltage)
hold on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%cpls
absCPLS_values = zeros(32, 1); % Assuming 33 buses, but we start from 2, so 32 values

% Initialize a vector to store the bus numbers
bus_numbers = 2:33; % Bus numbers from 2 to 33

for x = 2:33
    curvol = Voltage(x,1);
    curpowerreal = BD(x,2);
    curpowerim = BD(x,3);
    curR = LD(x-1,4);
    curX = LD(x-1,5);
    
    curCPLS = (2*curpowerreal*curR)/(curvol^2) + 1i*(2*curpowerim*curX)/(curvol^2);
    absCPLS = abs(curCPLS); % Calculate the absolute value
    
    % Store the absolute value in the vector
    absCPLS_values(x-1) = absCPLS;
end

% plot(absCPLS_values)
% Create a bar plot of the absolute values of curCPLS against the bus numbers

[sorted_absCPLS_values, sorted_indices] = sort(absCPLS_values, 'descend');

%plot(sorted_absCPLS_values)

% Determine the number of buses to select (half of the total buses)
num_buses_to_select = length(sorted_absCPLS_values) / 2;

% Select the first half of the sorted array
%selected_buses = sorted_indices(1:num_buses_to_select)
selected_buses = arrayfun(@(x) x + 1, sorted_indices(1:num_buses_to_select))

OBJ = calculateObjectiveFunction(Voltage, BD, LD, Ploss,Qloss);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%possible_tuples

% Define the range of sizes
no_of_dg_avail = 128;
sizes_range = linspace(1500/(1000*Sbase), 3000/(1000*Sbase), no_of_dg_avail);

% Initialize a cell array to store the tuples
bus_sizes_matrix = cell(length(selected_buses), length(sizes_range));

% Fill the cell array with tuples of bus numbers and sizes
for i = 1:length(selected_buses)
    for j = 1:length(sizes_range)
        % Create a tuple (bus number, size) and store it in the cell array
        bus_sizes_matrix{i, j} = [selected_buses(i), sizes_range(j)];
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



num_iter = 10;
num_agents = 64;


% Initialize an empty cell array to store the tuples
location_agent = cell(1, num_agents);

% Generate 50 tuples (i, j) with i and j being random integers between 1 and 16
for k = 1:num_agents
    i = randi([1, length(selected_buses)]); % Random integer between 1 and 16
    j = randi([1, no_of_dg_avail]); % Random integer between 1 and 16
    location_agent{k} = [i, j]; % Store the tuple in the cell array
end


%1-location, size,obj,agentnumber
alpha = zeros(4,1);
beta = zeros(4,1);
delta = zeros(4,1);

alpha(3)=intmax('int32');
beta(3)=intmax('int32');
delta(3)=intmax('int32');


for cur_agent = 1:num_agents
        % Assuming location_agent is a cell array where each cell contains a 2-element vector
        % representing the row and column indices in BD to update
        % location_agent{cur_agent}
        i = location_agent{cur_agent}(1);
        j = location_agent{cur_agent}(2);
        agent_loc_size = bus_sizes_matrix(i,j);
        bus_for_DG = agent_loc_size{1}(1);
        dg_size = agent_loc_size{1}(2);
        % dg_size = location_agent{cur_agent}(2);
        BD_temp = BD;
       % Store the old load power
        old_Pload = BD(bus_for_DG, 2);
     
        % Update the load power based on the new value
        % Assuming the new value is stored in the second column of location_agent
       BD_temp(bus_for_DG, 2) = old_Pload - dg_size;
       [Tl, TQl, Voltage, Vangle, Ploss, Qloss] = calculate_power_losses(LD,BD_temp,  Z, V, Sbase, num_iter);
       agent_obj =calculateObjectiveFunction(Voltage, BD_temp, LD, Ploss,Qloss);


        if agent_obj<alpha(3)
            alpha(1)=bus_for_DG;
            alpha(2)=dg_size;
            alpha(3)=agent_obj;
            alpha(4)=cur_agent;
        elseif agent_obj<beta(3)
            beta(1)=bus_for_DG;
            beta(2)=dg_size;
            beta(3)=agent_obj;
            beta(4)=cur_agent;
        elseif agent_obj<delta(3)
            delta(1)=bus_for_DG;
            delta(2)=dg_size;
            delta(3)=agent_obj;
            delta(4)=cur_agent;
        end
        
      % Optionally, you can print or store the old and new values for debugging
      % fprintf('Agent %d: Old Pload = %d, New Pload = %d\n', cur_agent, old_Pload, location_agent{cur_agent}(2));
    
end

% Initialize variables
a = 2;
final_loss = 0;
final_Voltage = Voltage;

% Loop through iterations
for cur_iter = 2:num_iter
    % Define Z_alpha, Z_beta, and Z_delta vectors
    Z_alpha = [alpha(1) alpha(2)];
    Z_beta = [beta(1) beta(2)];
    Z_delta = [delta(1) delta(2)];
    
    % Update 'a' variable linearly with every iteration
    a = a - a / num_iter;
    
    % Loop through agents
    for cur_agent = 1:num_agents
        % Skip alpha beta or gamma wolves
        if cur_agent == alpha(4) || cur_agent == beta(4) || cur_agent == delta(4)
            continue;
        end
        
        % Initialize random numbers for calculations
        r1 = rand;
        r2 = rand;
        C = 2 * r2;
        A = 2 * a * r1 - a;
        
        % Retrieve agent's current location and bus size
        i_t = location_agent{cur_agent}(1);
        j_t = location_agent{cur_agent}(2);
        agent_loc_size = bus_sizes_matrix(i_t, j_t);
        bus_for_DG = agent_loc_size{1}(1);
        dg_size = agent_loc_size{1}(2);
        
        % Calculate D vectors
        Z_t = [bus_for_DG dg_size];
        D_alpha = abs(C * Z_alpha - Z_t);
        D_beta = abs(C * Z_beta - Z_t);
        D_delta = abs(C * Z_delta - Z_t);
        
        %new positions following alpha,beta delta 
        Z_following_alpha = Z_alpha - A * D_alpha;
        Z_following_beta = Z_beta - A * D_beta;
        Z_following_delta = Z_delta - A * D_delta;
        
        % Search for closest positions in bus_sizes_matrix
        [Z_t_1_alpha, Z_t_1_alpha_indices] = search(bus_sizes_matrix, Z_following_alpha);
        [Z_t_1_beta, Z_t_1_beta_indices] = search(bus_sizes_matrix, Z_following_beta);
        [Z_t_1_delta, Z_t_1_delta_indices] = search(bus_sizes_matrix, Z_following_delta);
        
        % Calculate final position
        Z_final_temp = (Z_t_1_alpha + Z_t_1_beta + Z_t_1_delta) / 3;
        [Z_final, Z_final_indices] = search(bus_sizes_matrix, Z_final_temp);
        
        % GWO section
        % Retrieve new bus and DG sizes
        i_t_1 = Z_final_indices(1);
        j_t_1 = Z_final_indices(2);
        agent_loc_size = bus_sizes_matrix(i_t_1, j_t_1);
        bus_for_DG_GWO = agent_loc_size{1}(1);
        dg_size_GWO = agent_loc_size{1}(2);
        
        % Update load power and calculate losses
        BD_temp = BD;
        old_Pload = BD(bus_for_DG_GWO, 2);
        BD_temp(bus_for_DG_GWO, 2) = old_Pload - dg_size_GWO;
        [Tl, TQl, Voltage_GWO, Vangle, Ploss_GWO, Qloss_GWO] = calculate_power_losses(LD, BD_temp, Z, V, Sbase, num_iter);
        agent_obj_GWO = calculateObjectiveFunction(Voltage_GWO, BD_temp, LD, Ploss_GWO, Qloss_GWO);
        
        % DLH(dimension learning based hunting) section
        % Calculate new positions for DLH
        del_i = abs(i_t - i_t_1);
        del_j = abs(j_t - j_t_1);
        i_neigh_gwo = randi([abs(i_t - del_i), abs(i_t + del_i)]);
        j_neigh_gwo = randi([abs(j_t - del_j), abs(j_t + del_j)]);
        i_neigh_pop = randi([1, length(selected_buses)]);
        j_neigh_pop = randi([1, no_of_dg_avail]);
        
        % Ensure indices are within bounds
        Z_DLH_indices = [0 0];
        Z_DLH_indices(1) = randi([abs(i_t - abs(i_neigh_pop - i_neigh_gwo)), abs(i_t + abs(i_neigh_pop - i_neigh_gwo))]);
        Z_DLH_indices(2) = randi([abs(j_t - abs(j_neigh_pop - j_neigh_gwo)), abs(j_t + abs(j_neigh_pop - j_neigh_gwo))]);
        
        % Check bounds and adjust if necessary
        if Z_DLH_indices(1) > length(selected_buses)
            Z_DLH_indices(1) = length(selected_buses);
        end
        if Z_DLH_indices(2) > no_of_dg_avail
            Z_DLH_indices(2) = no_of_dg_avail;
        end
        if Z_DLH_indices(1) < 1
            Z_DLH_indices(1) = 1;
        end
        if Z_DLH_indices(2) < 1
            Z_DLH_indices(2) = 1;
        end
        
        % Retrieve new bus and DG sizes for DLH
        Z_DLH = bus_sizes_matrix(Z_DLH_indices(1), Z_DLH_indices(2));
        bus_for_DG_DLH = Z_DLH{1}(1);
        dg_size_DLH = Z_DLH{1}(2);
        
        % Update load power and calculate losses for DLH
        BD_temp = BD;
        old_Pload = BD(bus_for_DG_DLH, 2);
        BD_temp(bus_for_DG_DLH, 2) = old_Pload - dg_size_DLH;
        [Tl, TQl, Voltage_DLH, Vangle, Ploss_DLH, Qloss_DLH] = calculate_power_losses(LD, BD_temp, Z, V, Sbase, num_iter);
        agent_obj_DLH = calculateObjectiveFunction(Voltage_DLH, BD_temp, LD, Ploss_DLH, Qloss_DLH);
        
        % Determine the best objective function value between GWO and DLH
        agent_obj = agent_obj_GWO;
        bus_for_DG = bus_for_DG_GWO;
        dg_size = dg_size_GWO;
        location_agent{cur_agent} = Z_final_indices;
        Ploss = Ploss_GWO;
        Qloss = Qloss_GWO;
        Voltage = Voltage_GWO;
        
        if agent_obj_GWO > agent_obj_DLH
            agent_obj = agent_obj_DLH;
            bus_for_DG = bus_for_DG_DLH;
            dg_size = dg_size_DLH;
            location_agent{cur_agent} = Z_DLH_indices;
            Ploss = Ploss_DLH;
            Qloss = Qloss_DLH;
            Voltage = Voltage_DLH;
        end
        
        % Constraint check
        constraint_broken = 0;
        for i = 1:33
            if Voltage(i) > 1.05 || Voltage(i) < 0.9
                constraint_broken = 1;
            end
        end
        
        % Skip iteration if constraint is broken
        if constraint_broken == 1
            continue;
        end
        
        % Update final positions and objective values
        location_agent{cur_agent} = Z_final_indices;
        
        % Update alpha, beta, and delta based on objective values
        if agent_obj < alpha(3)
            alpha(1) = bus_for_DG;
            alpha(2) = dg_size;
            alpha(3) = agent_obj;
            alpha(4) = cur_agent;
            complexVector = sum(Ploss) + 1i * sum(Qloss); % Combine real and imaginary parts
            final_loss = norm(complexVector); % Calculate the magnitude
            final_Voltage = Voltage;
        elseif agent_obj < beta(3)
            beta(1) = bus_for_DG;
            beta(2) = dg_size;
            beta(3) = agent_obj;
            beta(4) = cur_agent;
        elseif agent_obj < delta(3)
            delta(1) = bus_for_DG;
            delta(2) = dg_size;
            delta(3) = agent_obj;
            delta(4) = cur_agent;
        end
    end
end

selected_buses
alpha
beta
delta
initial_loss
final_loss
Loss_reduction_percentage = ((initial_loss-final_loss)*100)/initial_loss
% Plot final_Voltage on the first figure
% Creates a new figure window
plot(final_Voltage);
xlabel('Bus Number');
ylabel('Voltage(pu)');
title('Voltage Profile');
grid on;

% Create a new figure window for the bar plot
figure; % Creates another new figure window
bar(bus_numbers, absCPLS_values);
xlabel('Bus Number');
ylabel('Absolute Value of CPLS');
title('Absolute Values of CPLS vs. Bus Number');
grid on;




% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

function [Z_t_1_alpha, Z_t_1_alpha_indices] = search(bus_sizes_matrix, Z_following_alpha)
    % Initialize variables to store the closest match and its index
    closestMatch = [];
    closestIndex = [];
    minDifference = inf; % Initialize with infinity to ensure any first comparison will be smaller
    
    % Iterate through each cell in bus_sizes_matrix
    for i = 1:size(bus_sizes_matrix, 1)
        for j = 1:size(bus_sizes_matrix, 2)
            % Extract the numeric array from the cell
            currentElement = bus_sizes_matrix{i, j};
            
            % Calculate the difference between the current element and Z_following_alpha
            difference = norm(currentElement - Z_following_alpha);
            
            % Check if the current difference is smaller than the previous minimum difference
            if difference < minDifference
                % Update the closest match and its index
                closestMatch = currentElement;
                closestIndex = [i, j];
                minDifference = difference;
            end
        end
    end
    
    % Return the closest match and its index
    Z_t_1_alpha = closestMatch;
    Z_t_1_alpha_indices = closestIndex;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%objective function

function OBJ = calculateObjectiveFunction(Voltage, BD, LD, Ploss,Qloss)
    % TPL = 0;
    TVD = 0;
    VSI = 0;
    for x = 2:33
        curvol = Voltage(x,1);
        curpowerreal = BD(x,2);
        curpowerim = BD(x,3);
        curR = LD(x-1,4);
        curX = LD(x-1,5);
        
        % TPL = TPL + Ploss(x-1,1);

        TVD = TVD + ((curvol - Voltage(1,1))^2);

        VSI = VSI + curvol^4 - (4*(curpowerreal*curR + curpowerim*curX)*curvol^2) - (4*(curpowerreal*curX + curpowerim*curR));
    end
    complexVector = sum(Ploss) + 1i*sum(Qloss); % Combine real and imaginary parts
    s_loss = norm(complexVector); 
    F1 = s_loss;
    F2 = TVD;
    F3 = 1/VSI;
    R1 =  0.8;
    R2 = 0.1;
    R3 = 0.1;
    OBJ = F1*R1 + F2*R2 + F3*R3;
end


function [Tl, TQl, Voltage, Vangle, Ploss, Qloss] = calculate_power_losses(LD,BD,  Z, V, Sbase, Iter)
    % MAXIMUM NUMBER OF ITERATIONS FOR BACKWARD-FORWARD SWEEP ALGORITHM
    % LD: Load data matrix
    % Z: Line impedance vector
    % V: Initial voltage vector
    % Sbase: System base power
    % Iter: Maximum number of iterations
    
    Sload=complex(BD(:,2),BD(:,3)); % COMPLEX LOAD IN PER UNIT VALUES
    % Initialize variables
    Iload = zeros(size(V)); % Assuming Iload is initialized to zero
    Iline = zeros(size(LD, 1), 1); % Initialize line current vector

    for i = 1:Iter
        % STARTING WITH BACKWARD SWEEP FOR LINE CURRENT CALCULATIONS
        Iload = conj(Sload./V);
        for j = size(LD,1):-1:1   % STARTING FROM THE END OF THE FEEDER
            c = [];
            e = [];
            [c e] = find(LD(:,2:3)==LD(j,3));
            if size(c,1)==1     % IF SIZE(C,1) IS ONE THAN BUS "J" IS STARTING OR ENDING BUS
                Iline(LD(j,1)) = Iload(LD(j,3));
            else
                Iline(LD(j,1)) = Iload(LD(j,3)) + sum(Iline(LD(c,1))) - Iline(LD(j,1));
            end
        end
        % STARTING THE FORWARD SWEEP FOR BUS-VOLTAGE CALCULATION
        for j = 1:size(LD,1)
            V(LD(j,3)) = V(LD(j,2)) - Iline(LD(j,1))*Z(j);
        end
    end

    % Calculate power losses and voltages
    Voltage = abs(V);
    Vangle = angle(V);
    Ploss = real(Z.*(abs(Iline.^2)));
    Qloss = imag(Z.*(abs(Iline.^2)));
    Tl = sum(Ploss)*Sbase*1000;
    TQl = sum(Qloss)*Sbase*1000;
end


     