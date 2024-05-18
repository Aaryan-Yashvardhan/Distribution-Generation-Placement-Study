nbus=69;
%index, starting bus, ending bus, R, X
LD=[1	1	2	0.0005	0.0012
    2	2	3	0.0005	0.0012
    3	3	4	0.0015	0.0036
    4	4	5	0.0251	0.0294
    5	5	6	0.366	0.1864
    6	6	7	0.3811	0.1941
    7	7	8	0.0922	0.047
    8	8	9	0.0493	0.0251
    9	9	10	0.819	0.2707
    10	10	11	0.1872	0.0619
    11	11	12	0.7114	0.2351
    12	12	13	1.03	0.34
    13	13	14	1.044	0.345
    14	14	15	1.058	0.3496
    15	15	16	0.1966	0.065
    16	16	17	0.3744	0.1238
    17	17	18	0.0047	0.0016
    18	18	19	0.3276	0.1083
    19	19	20	0.2106	0.069
    20	20	21	0.3416	0.1129
    21	21	22	0.014	0.0046
    22	22	23	0.1591	0.0526
    23	23	24	0.3463	0.1145
    24	24	25	0.7488	0.2475
    25	25	26	0.3089	0.1021
    26	26	27	0.1732	0.0572
    27	3	28	0.0044	0.0108
    28	28	29	0.064	0.1565
    29	29	30	0.3978	0.1315
    30	30	31	0.0702	0.0232
    31	31	32	0.351	0.116
    32	32	33	0.839	0.2816
    33	33	34	1.708	0.5646
    34	34	35	1.474	0.4873
    35	3	36	0.0044	0.0108
    36	36	37	0.064	0.1565
    37	37	38	0.1053	0.123
    38	38	39	0.0304	0.0355
    39	39	40	0.0018	0.0021
    40	40	41	0.7283	0.8509
    41	41	42	0.31	0.3623
    42	42	43	0.041	0.0478
    43	43	44	0.0092	0.0116
    44	44	45	0.1089	0.1373
    45	45	46	0.0009	0.0012
    46	4	47	0.0034	0.0084
    47	47	48	0.0851	0.2083
    48	48	49	0.2898	0.7091
    49	49	50	0.0822	0.2011
    50	8	51	0.0928	0.0473
    51	51	52	0.3319	0.1114
    52	9	53	0.174	0.0886
    53	53	54	0.203	0.1034
    54	54	55	0.2842	0.1447
    55	55	56	0.2813	0.1433
    56	56	57	1.59	0.5337
    57	57	58	0.7837	0.263
    58	58	59	0.3042	0.1006
    59	59	60	0.3861	0.1172
    60	60	61	0.5075	0.2585
    61	61	62	0.0974	0.0496
    62	62	63	0.145	0.0738
    63	63	64	0.7105	0.3619
    64	64	65	1.041	0.5302
    65	11	66	0.2012	0.0611
    66	66	67	0.0047	0.0014
    67	12	68	0.7394	0.2444
    68	68	69	0.0047	0.0016];

%bus no, Pload, Qload

BD=[1	0	0	
2	0	0	
3	0	0	
4	0	0	
5	0	0	
6	2.6	2.2	
7	40.4	30	
8	75	54	
9	30	22	
10	28	19	
11	145	104	
12	145	104	
13	8	5	
14	8	5.5	
15	0	0	
16	45.5	30	
17	60	35	
18	60	35	
19	0	0	
20	1	0.6	
21	114	81	
22	5	3.5	
23	0	0	
24	28	20	
25	0	0	
26	14	10	
27	14	10	
28	26	18.6	
29	26	18.6	
30	0	0	
31	0	0	
32	0	0	
33	14	10	
34	19.5	14	
35	6	4	
36	26	18.55	
37	26	18.55	
38	0	0	
39	24	17	
40	24	17	
41	1.2	1	
42	0	0	
43	6	4.3	
44	0	0	
45	39.22	26.3	
46	39.22	26.3	
47	0	0	
48	79	56.4	
49	384.7	274.5	
50	384.7	274.5	
51	40.5	28.3	
52	3.6	2.7	
53	4.35	3.5	
54	26.4	19	
55	24	17.2	
56	0	0	
57	0	0	
58	0	0	
59	100	72	
60	0	0	
61	1244	888	
62	32	23	
63	0	0	
64	227	162	
65	59	42	
66	18	13	
67	18	13	
68	28	20	
69	28	20	];



% Base MVA and KV values
Sbase = 100;                  % BASE MVA VALUE
Vbase = 12.66;                % BASE KV VALUE

% Base impedance value
Zbase = (Vbase^2) / Sbase;

% Convert line data to per unit values
LD(:, 4:5) = LD(:, 4:5) / Zbase;  

% Convert load data to per unit values
BD(:, 2:3) = BD(:, 2:3) / (1000 * Sbase);  

% Maximum number of buses
N = max(max(LD(:, 2:3)));

% Initializations
V = ones(size(BD, 1), 1);                    % INITIAL VALUE OF VOLTAGES
Z = complex(LD(:, 4), LD(:, 5));             % LINE IMPEDANCE IN PER UNIT VALUES
Iline = zeros(size(LD, 1), 1);               % LINE CURRENT MATRIX

% Calculate power losses and voltages
[Tl, TQl, Voltage, Vangle, Ploss, Qloss] = calculate_power_losses(LD, BD, Z, V, Sbase, 200);
complexVector = sum(Ploss) + 1i * sum(Qloss); % Combine real and imaginary parts
initial_loss = norm(complexVector);           % Calculate the magnitude

% Plot initial voltages
plot(Voltage)
hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate CPLS and select buses

absCPLS_values = zeros(68, 1); % Assuming 69 buses, but we start from 2, so 68 values
bus_numbers = 2:69; % Bus numbers from 2 to 69

% Loop through buses to calculate CPLS
for x = 2:69
    curvol = Voltage(x, 1);
    curpowerreal = BD(x, 2);
    curpowerim = BD(x, 3);
    curR = LD(x-1, 4);
    curX = LD(x-1, 5);
    
    curCPLS = (2 * curpowerreal * curR) / (curvol^2) + 1i * (2 * curpowerim * curX) / (curvol^2);
    absCPLS = abs(curCPLS); % Calculate the absolute value
    
    % Store the absolute value in the vector
    absCPLS_values(x-1) = absCPLS;
end

% Sort CPLS values
[sorted_absCPLS_values, sorted_indices] = sort(absCPLS_values, 'descend');

% Determine the number of buses to select
num_buses_to_select = (length(sorted_absCPLS_values) / 2);

% Select the first half of the sorted array
selected_buses = arrayfun(@(x) x + 1, sorted_indices(1:num_buses_to_select));

% Calculate the objective function
OBJ = calculateObjectiveFunction(Voltage, BD, LD, Ploss, Qloss);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create tuples for possible combinations

% Define the range of sizes
no_of_dg_avail = 30;
sizes_range_P = linspace(1500 / (1000 * Sbase), 3000 / (1000 * Sbase), no_of_dg_avail);
sizes_range_Q = linspace(1500 / (1000 * Sbase), 3000 / (1000 * Sbase), no_of_dg_avail);

% Initialize a cell array to store the tuples
bus_sizes_matrix = cell(length(selected_buses), length(sizes_range_P), length(sizes_range_Q));

% Fill the cell array with tuples of bus numbers and sizes
for i = 1:length(selected_buses)
    for j = 1:length(sizes_range_P)
        for k = 1:length(sizes_range_Q)
            % Create a tuple (bus number, size) and store it in the cell array
            bus_sizes_matrix{i, j, k} = [selected_buses(i), sizes_range_P(j), sizes_range_Q(k)];
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Other initialization
num_iter = 10;
num_agents = 64;

% Initialize an empty cell array to store the tuples
location_agent = cell(1, num_agents);

% Generate tuples with random integers
for cur = 1:num_agents
    i = randi([1, length(selected_buses)]); % Random integer between 1 and 16
    j = randi([1, no_of_dg_avail]); % Random integer between 1 and 16
    k = randi([1, no_of_dg_avail]); % Random integer between 1 and 16
    location_agent{cur} = [i, j, k]; % Store the tuple in the cell array
end

% Initialize alpha, beta, delta vectors
alpha = zeros(5, 1);
beta = zeros(5, 1);
delta = zeros(5, 1);

alpha(4) = intmax('int32');
beta(4) = intmax('int32');
delta(4) = intmax('int32');






% Iterate through each agent
for cur_agent = 1:num_agents
    % Extract the indices from the current agent's location
    i = location_agent{cur_agent}(1);
    j = location_agent{cur_agent}(2);
    k = location_agent{cur_agent}(3);
    
    % Get the size of the distributed generator from bus_sizes_matrix
    agent_loc_size = bus_sizes_matrix(i, j, k);
    bus_for_DG = agent_loc_size{1}(1);
    dg_size_p = agent_loc_size{1}(2);
    dg_size_q = agent_loc_size{1}(3);
    
    % Store the old load power
    old_Pload = BD(bus_for_DG, 2);
    old_Qload = BD(bus_for_DG, 3);
    
    % Update the load power based on the new value
    BD_temp = BD;
    BD_temp(bus_for_DG, 2) = old_Pload - dg_size_p;
    BD_temp(bus_for_DG, 3) = old_Qload - dg_size_q;
    
    % Calculate power losses and voltages with the updated load data
    [Tl, TQl, Voltage, Vangle, Ploss, Qloss] = calculate_power_losses(LD, BD_temp, Z, V, Sbase, num_iter);
    
    % Calculate the objective function value for the agent's configuration
    agent_obj = calculateObjectiveFunction(Voltage, BD_temp, LD, Ploss, Qloss);

    % Update alpha, beta, delta based on the objective function value
    if agent_obj < alpha(4)
        alpha(1) = bus_for_DG;
        alpha(2) = dg_size_p;
        alpha(3) = dg_size_q;
        alpha(4) = agent_obj;
        alpha(5) = cur_agent;
    elseif agent_obj < beta(4)
        beta(1) = bus_for_DG;
        beta(2) = dg_size_p;
        beta(3) = dg_size_q;
        beta(4) = agent_obj;
        beta(5) = cur_agent;
    elseif agent_obj < delta(4)
        delta(1) = bus_for_DG;
        delta(2) = dg_size_p;
        delta(3) = dg_size_q;
        delta(4) = agent_obj;
        delta(5) = cur_agent;
    end
    
    % Optionally, print or store old and new values for debugging
    % fprintf('Agent %d: Old Pload = %d, New Pload = %d\n', cur_agent, old_Pload, location_agent{cur_agent}(2));
end

% Initialize a variable to represent the current iteration
a = 2;

% Initialize the final loss variable
final_loss = 0;
final_Voltage = Voltage;

% Loop through iterations
for cur_iter = 2:num_iter
    % Extract alpha, beta, and delta values
    Z_alpha = [alpha(1), alpha(2), alpha(3)];
    Z_beta = [beta(1), beta(2), beta(3)];
    Z_delta = [delta(1), delta(2), delta(3)];
    
    % Update variable a
    a = a - a / num_iter;
    
    % Loop through agents
    for cur_agent = 1:num_agents
        % Skip alpha, beta, and delta agents
        if cur_agent == alpha(5) || cur_agent == beta(5) || cur_agent == delta(5)
            continue;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Initial position
        
        % Generate random values
        r1 = rand;
        r2 = rand;
        C = 2 * r2;
        A = 2 * a * r1 - a;

        % Extract current agent's position
        i_t = location_agent{cur_agent}(1);
        j_t = location_agent{cur_agent}(2);
        k_t = location_agent{cur_agent}(3);
        agent_loc_size = bus_sizes_matrix(i_t, j_t, k_t);
        bus_for_DG = agent_loc_size{1}(1);
        dg_size_p = agent_loc_size{1}(2);
        dg_size_q = agent_loc_size{1}(3);
        Z_t = [bus_for_DG, dg_size_p, dg_size_q];

        % Calculate differences
        D_alpha = abs(C * Z_alpha - Z_t);
        D_beta = abs(C * Z_beta - Z_t);
        D_delta = abs(C * Z_delta - Z_t);

        % Calculate new positions
        Z_following_alpha = abs(Z_alpha - A * D_alpha);
        Z_following_beta = abs(Z_beta - A * D_beta);
        Z_following_delta = abs(Z_delta - A * D_delta);
        
        % Search for nearest positions
        [Z_t_1_alpha, Z_t_1_alpha_indices] = search(bus_sizes_matrix, Z_following_alpha);
        [Z_t_1_beta, Z_t_1_beta_indices] = search(bus_sizes_matrix, Z_following_beta);
        [Z_t_1_delta, Z_t_1_delta_indices] = search(bus_sizes_matrix, Z_following_delta);

        % Calculate average position
        Z_final_temp = (Z_t_1_alpha + Z_t_1_beta + Z_t_1_delta) / 3;
        [Z_final, Z_final_indices] = search(bus_sizes_matrix, Z_final_temp);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GWO
        
        % Extract GWO position
        i_t_1 = Z_final_indices(1);
        j_t_1 = Z_final_indices(2);
        k_t_1 = Z_final_indices(3);
        agent_loc_size = bus_sizes_matrix(i_t_1, j_t_1, k_t_1);
        bus_for_DG_GWO = agent_loc_size{1}(1);
        dg_size_GWO_p = agent_loc_size{1}(2);
        dg_size_GWO_q = agent_loc_size{1}(3);

        % Update load power
        BD_temp = BD;
        old_Pload = BD(bus_for_DG_GWO, 2);
        old_Qload = BD(bus_for_DG_GWO, 3);
        BD_temp(bus_for_DG_GWO, 2) = old_Pload - dg_size_GWO_p;
        BD_temp(bus_for_DG_GWO, 3) = old_Qload - dg_size_GWO_q;
        
        % Calculate power losses and voltage
        [Tl, TQl, Voltage_GWO, Vangle, Ploss_GWO, Qloss_GWO] = calculate_power_losses(LD, BD_temp, Z, V, Sbase, num_iter);
        agent_obj_GWO = calculateObjectiveFunction(Voltage_GWO, BD_temp, LD, Ploss_GWO, Qloss_GWO);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % DLH
        
        % Calculate differences in indices
        del_i = abs(i_t - i_t_1);
        del_j = abs(j_t - j_t_1);
        del_k = abs(k_t - k_t_1);

        % Generate random neighboring indices
        i_neigh_gwo = randi([abs(i_t - del_i), abs(i_t + del_i)]);
        j_neigh_gwo = randi([abs(j_t - del_j), abs(j_t + del_j)]);
        k_neigh_gwo = randi([abs(k_t - del_k), abs(k_t + del_k)]);
        i_neigh_pop = randi([1, length(selected_buses)]);
        j_neigh_pop = randi([1, no_of_dg_avail]);
        k_neigh_pop = randi([1, no_of_dg_avail]);
        
        % Calculate DLH indices
        Z_DLH_indices = [0, 0, 0];
        Z_DLH_indices(1) = randi([abs(i_t - abs(i_neigh_pop - i_neigh_gwo)), abs(i_t + abs(i_neigh_pop - i_neigh_gwo))]);
        Z_DLH_indices(2) = randi([abs(j_t - abs(j_neigh_pop - j_neigh_gwo)), abs(j_t + abs(j_neigh_pop - j_neigh_gwo))]);
        Z_DLH_indices(3) = randi([abs(k_t - abs(k_neigh_pop - k_neigh_gwo)), abs(k_t + abs(k_neigh_pop - k_neigh_gwo))]);
        
        % Check boundary conditions
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
        if Z_DLH_indices(3) < 1
            Z_DLH_indices(3) = 1;
        end
        if Z_DLH_indices(3) > no_of_dg_avail
            Z_DLH_indices(3) = no_of_dg_avail;
        end
        
        % Extract DLH position
        Z_DLH = bus_sizes_matrix(Z_DLH_indices(1), Z_DLH_indices(2), Z_DLH_indices(3));
        bus_for_DG_DLH = Z_DLH{1}(1);
        dg_size_DLH_p = Z_DLH{1}(2);
        dg_size_DLH_q = Z_DLH{1}(3);

        % Update load power
        BD_temp = BD;
        old_Pload = BD(bus_for_DG_DLH, 2);
        old_Qload = BD(bus_for_DG_DLH, 3);
        BD_temp(bus_for_DG_DLH, 2) = old_Pload - dg_size_DLH_p;
        BD_temp(bus_for_DG_DLH, 3) = old_Qload - dg_size_DLH_q;
        
        % Calculate power losses and voltage
        [Tl, TQl, Voltage_DLH, Vangle, Ploss_DLH, Qloss_DLH] = calculate_power_losses(LD, BD_temp, Z, V, Sbase, num_iter);
        agent_obj_DLH = calculateObjectiveFunction(Voltage_DLH, BD_temp, LD, Ploss_DLH, Qloss_DLH);

        % Choose between GWO and DLH
        agent_obj = agent_obj_GWO;
        bus_for_DG = bus_for_DG_GWO;
        dg_size_p = dg_size_GWO_p;
        dg_size_q = dg_size_GWO_q;
        location_agent{cur_agent} = Z_final_indices;
        Ploss = Ploss_GWO;
        Qloss = Qloss_GWO;
        Voltage = Voltage_GWO;

        if agent_obj_GWO > agent_obj_DLH
            agent_obj = agent_obj_DLH;
            bus_for_DG = bus_for_DG_DLH;
            dg_size_p = dg_size_DLH_p;
            dg_size_q = dg_size_DLH_q;
            location_agent{cur_agent} = Z_DLH_indices;
            Ploss = Ploss_DLH;
            Qloss = Qloss_DLH;
            Voltage = Voltage_DLH;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Constraint

        % Check voltage constraint
        constraint_broken = 0;
        for i = 1:69
            if Voltage(i) > 1.1 || Voltage(i) < 0.9
                constraint_broken = 1;
            end
        end
        
        % If constraint broken, continue to the next iteration
        if constraint_broken == 1
            continue;
        end

        location_agent{cur_agent} = Z_final_indices;

        % Update alpha, beta, and delta if necessary
        if agent_obj < alpha(4)
            alpha(1) = bus_for_DG;
            alpha(2) = dg_size_p;
            alpha(3) = dg_size_q;
            alpha(4) = agent_obj;
            alpha(5) = cur_agent;
            complexVector = sum(Ploss) + 1i * sum(Qloss); % Combine real and imaginary parts
            final_loss = norm(complexVector); % Calculate the magnitude
            final_Voltage = Voltage;
        elseif agent_obj < beta(4)
            beta(1) = bus_for_DG;
            beta(2) = dg_size_p;
            beta(3) = dg_size_q;
            beta(4) = agent_obj;
            beta(5) = cur_agent;
        elseif agent_obj < delta(4)
            delta(1) = bus_for_DG;
            delta(2) = dg_size_p;
            delta(3) = dg_size_q;
            delta(4) = agent_obj;
            delta(5) = cur_agent;
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
plot(final_Voltage)
xlabel('Bus Number');
ylabel('Voltage(pu)');
title('Voltage Profile');
grid on;
figure;
bar(bus_numbers, absCPLS_values);
xlabel('Bus Number');
ylabel('Absolute Value of CPLS');
title('Absolute Values of CPLS vs. Bus Number');
grid on;


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Search function to find the closest match to Z_following_alpha in bus_sizes_matrix
function [Z_t_1_alpha, Z_t_1_alpha_indices] = search(bus_sizes_matrix, Z_following_alpha)
    % Initialize variables to store the closest match and its index
    closestMatch = [];
    closestIndex = [];
    minDifference = inf; % Initialize with infinity to ensure any first comparison will be smaller
    
    % Iterate through each cell in bus_sizes_matrix
    for i = 1:size(bus_sizes_matrix, 1)
        for j = 1:size(bus_sizes_matrix, 2)
            for k = 1:size(bus_sizes_matrix, 3)
                % Extract the numeric array from the cell
                currentElement = bus_sizes_matrix{i, j, k};
            
                % Calculate the difference between the current element and Z_following_alpha
                difference = norm(currentElement - Z_following_alpha);
            
                % Check if the current difference is smaller than the previous minimum difference
                if difference < minDifference
                    % Update the closest match and its index
                    closestMatch = currentElement;
                    closestIndex = [i, j, k];
                    minDifference = difference;
                end
            end
        end
    end
    
    % Return the closest match and its index
    Z_t_1_alpha = closestMatch;
    Z_t_1_alpha_indices = closestIndex;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Objective function to calculate the objective value based on given parameters

function OBJ = calculateObjectiveFunction(Voltage, BD, LD, Ploss, Qloss)
    % Initialize variables for Total Voltage Deviation (TVD) and Voltage Stability Index (VSI)
    TVD = 0;
    VSI = 0;
    
    % Loop through each bus except the slack bus
    for x = 2:69
        % Extract voltage, active power, and reactive power for the current bus
        curvol = Voltage(x, 1);
        curpowerreal = BD(x, 2);
        curpowerim = BD(x, 3);
        curR = LD(x - 1, 4);
        curX = LD(x - 1, 5);

        % Calculate Total Voltage Deviation (TVD)
        TVD = TVD + ((curvol - Voltage(1, 1))^2);

        % Calculate Voltage Stability Index (VSI)
        VSI = VSI + curvol^4 - (4 * (curpowerreal * curR + curpowerim * curX) * curvol^2) - (4 * (curpowerreal * curX + curpowerim * curR));
    end
    
    % Calculate total system loss
    complexVector = sum(Ploss) + 1i * sum(Qloss); % Combine real and imaginary parts
    s_loss = norm(complexVector); 
    
    % Calculate objective function value using weighted sum of TVD, VSI, and system loss
    F1 = s_loss;
    F2 = TVD;
    F3 = 1 / VSI;
    R1 =  1 ;
    R2 = 0;
    R3 = 0;
    OBJ = F1 * R1 + F2 * R2 + F3 * R3;
end

% Function to calculate power losses and voltages using backward-forward sweep algorithm
function [Tl, TQl, Voltage, Vangle, Ploss, Qloss] = calculate_power_losses(LD, BD, Z, V, Sbase, Iter)
    % Initialize load data, line impedance, voltage, and current variables
    Sload = complex(BD(:, 2), BD(:, 3));
    Iload = zeros(size(V));
    Iline = zeros(size(LD, 1), 1);

    % Perform backward-forward sweep algorithm for Iter iterations
    for i = 1:Iter
        % Backward sweep
        Iload = conj(Sload ./ V);
        for j = size(LD, 1):-1:1
            [c, e] = find(LD(:, 2:3) == LD(j, 3));
            if size(c, 1) == 1
                Iline(LD(j, 1)) = Iload(LD(j, 3));
            else
                Iline(LD(j, 1)) = Iload(LD(j, 3)) + sum(Iline(LD(c, 1))) - Iline(LD(j, 1));
            end
        end
        
        % Forward sweep
        for j = 1:size(LD, 1)
            V(LD(j, 3)) = V(LD(j, 2)) - Iline(LD(j, 1)) * Z(j);
        end
    end

    % Calculate power losses and voltages
    Voltage = abs(V);
    Vangle = angle(V);
    Ploss = real(Z .* (abs(Iline.^2)));
    Qloss = imag(Z .* (abs(Iline.^2)));
    Tl = sum(Ploss) * Sbase * 1000;
    TQl = sum(Qloss) * Sbase * 1000;
end

     