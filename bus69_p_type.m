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
    68	68	69	0.0047	0.0016 ];

%bus no, Pload, Qload
BD=[1	0	0	0
2	0	0	0
3	0	0	0
4	0	0	0
5	0	0	0
6	2.6	2.2	0
7	40.4	30	0
8	75	54	0
9	30	22	0
10	28	19	0
11	145	104	0
12	145	104	0
13	8	5	0
14	8	5.5	0
15	0	0	0
16	45.5	30	0
17	60	35	0
18	60	35	0
19	0	0	0
20	1	0.6	0
21	114	81	0
22	5	3.5	0
23	0	0	0
24	28	20	0
25	0	0	0
26	14	10	0
27	14	10	0
28	26	18.6	0
29	26	18.6	0
30	0	0	0
31	0	0	0
32	0	0	0
33	14	10	0
34	19.5	14	0
35	6	4	0
36	26	18.55	0
37	26	18.55	0
38	0	0	0
39	24	17	0
40	24	17	0
41	1.2	1	0
42	0	0	0
43	6	4.3	0
44	0	0	0
45	39.22	26.3	0
46	39.22	26.3	0
47	0	0	0
48	79	56.4	0
49	384.7	274.5	0
50	384.7	274.5	0
51	40.5	28.3	0
52	3.6	2.7	0
53	4.35	3.5	0
54	26.4	19	0
55	24	17.2	0
56	0	0	0
57	0	0	0
58	0	0	0
59	100	72	0
60	0	0	0
61	1244	888	0
62	32	23	0
63	0	0	0
64	227	162	0
65	59	42	0
66	18	13	0
67	18	13	0
68	28	20	0
69	28	20	0];



% Define base values
Sbase=100;                  % BASE MVA VALUE
Vbase=12.66;                   % BASE KV VALUE
Zbase=(Vbase^2)/Sbase;      % BASE IMPEDANCE VALUE

% Convert values of line resistance and reactance to per unit values
LD(:,4:5)=LD(:,4:5)/Zbase;  % PER UNIT VALUES OF LINE R AND X

% Convert values of loads P and Q at each bus to per unit values
BD(:,2:3)=BD(:,2:3)/(1000*Sbase);  % PER UNIT VALUES LOADS P AND Q AT EACH BUS

% Find the maximum value across the matrices LD and BD
N=max(max(LD(:,2:3)));

% Initialize voltage vector with ones
V=ones(size(BD,1),1);                    % INITIAL VALUE OF VOLTAGES

% Calculate line impedance in per unit values
Z=complex(LD(:,4),LD(:,5));     % LINE IMPEDANCE IN PER UNIT VALUES

% Initialize line current matrix with zeros
Iline=zeros(size(LD,1),1);      % LINE CURRENT MATRIX

% Calculate power losses
[Tl, TQl, Voltage, Vangle, Ploss, Qloss] = calculate_power_losses(LD,BD,  Z, V, Sbase, 200);
% Assuming Ploss and Qloss are vectors of the same length
complexVector = sum(Ploss) + 1i*sum(Qloss); % Combine real and imaginary parts
initial_loss = norm(complexVector); % Calculate the magnitude

% vector for the absolute values of CPLS for each bus
absCPLS_values = zeros(68, 1); % Assuming 69 buses, but we start from 2, so 68 values

% Initialize a vector to store the bus numbers
bus_numbers = 2:69; % Bus numbers from 2 to 69

for x = 2:69
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

% Plot the absolute values of CPLS against the bus numbers


% Sort the absolute values of CPLS in descending order
[sorted_absCPLS_values, sorted_indices] = sort(absCPLS_values, 'descend');

% Determine the number of buses to select (half of the total buses)
num_buses_to_select = length(sorted_absCPLS_values) / 2;

% Select the first half of the sorted array
selected_buses = arrayfun(@(x) x + 1, sorted_indices(1:num_buses_to_select));

% Calculate the objective function
OBJ = calculateObjectiveFunction(Voltage, BD, LD, Ploss,Qloss);
plot(Voltage);
hold on;
% Define the range of sizes for DG units
no_of_dg_avail = 128;
sizes_range = linspace(1500/(1000*Sbase), 3000/(1000*Sbase), no_of_dg_avail);

% Initialize a cell array to store the tuples of bus numbers and sizes
bus_sizes_matrix = cell(length(selected_buses), length(sizes_range));

% Fill the cell array with tuples of bus numbers and sizes
for i = 1:length(selected_buses)
    for j = 1:length(sizes_range)
        % Create a tuple (bus number, size) and store it in the cell array
        bus_sizes_matrix{i, j} = [selected_buses(i), sizes_range(j)];
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5


% Set the number of iterations and number of agents
num_iter = 20;
num_agents = 64;

% Initialize an empty cell array to store the tuples
location_agent = cell(1, num_agents);

% Generate random tuples (i, j)
for k = 1:num_agents
    i = randi([1, length(selected_buses)]); % Random integer between 1 and number of selected buses 
    j = randi([1, no_of_dg_avail]); % Random integer between 1 and no. of DGs available
    location_agent{k} = [i, j]; % Store the tuple in the cell array
end



% Initialize variables for alpha, beta, and delta
%location of dg, dg size, objective function, agent number
alpha = zeros(4,1);
beta = zeros(4,1);
delta = zeros(4,1);

% Initialize objective function values to maximum integer value
alpha(3)=intmax('int32');
beta(3)=intmax('int32');
delta(3)=intmax('int32');

% Loop through agents to evaluate objective function and update alpha, beta, delta
for cur_agent = 1:num_agents
        % Extract location and size of the agent's DG
        i = location_agent{cur_agent}(1);
        j = location_agent{cur_agent}(2);
        agent_loc_size = bus_sizes_matrix(i,j);
        bus_for_DG = agent_loc_size{1}(1);
        dg_size = agent_loc_size{1}(2);
        BD_temp = BD;
        old_Pload = BD(bus_for_DG, 2); % Store the old load power
     
        % Update the load power based on the new value
        BD_temp(bus_for_DG, 2) = old_Pload - dg_size;
        
        % Calculate power losses and objective function
        [Tl, TQl, Voltage, Vangle, Ploss, Qloss] = calculate_power_losses(LD,BD_temp,  Z, V, Sbase, num_iter);
        agent_obj = calculateObjectiveFunction(Voltage, BD_temp, LD, Ploss,Qloss);

        % Update alpha, beta, delta based on objective function value
        if agent_obj < alpha(3)
            alpha(1) = bus_for_DG;
            alpha(2) = dg_size;
            alpha(3) = agent_obj;
            alpha(4) = cur_agent;
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
        
        % Optionally, print or store the old and new values for debugging
        % fprintf('Agent %d: Old Pload = %d, New Pload = %d\n', cur_agent, old_Pload, location_agent{cur_agent}(2));
end

a=2; % Initialize a constant value 'a'

final_loss=0; % Initialize final_loss (the final power loss alpha wolf incurs) variable to zero
final_Voltage=Voltage; % Initialize final_Voltage variable to the current Voltage value

% Loop through iterations
for cur_iter=1:num_iter
    
    % Extract values from alpha, beta, delta
    Z_alpha = [alpha(1) alpha(2)];
    Z_beta = [beta(1) beta(2)];
    Z_delta = [delta(1) delta(2)];
    
    % Update 'a' value
    a=a-a/num_iter;
    
    % Loop through agents
    for cur_agent=1:num_agents
        
        % Skip alpha, beta, delta agents
        if cur_agent==alpha(4) ||  cur_agent==beta(4) || cur_agent==delta(4)
            continue;
        end

        % Initial position calculations
        
        % Generate random numbers
        r1 = rand;
        r2 = rand;
        C=2*r2;
        A=2*a*r1 - a;

        % Extract current location and size of the agent's DG
        i_t = location_agent{cur_agent}(1);
        j_t = location_agent{cur_agent}(2);
        agent_loc_size = bus_sizes_matrix(i_t,j_t);
        bus_for_DG = agent_loc_size{1}(1);
        dg_size = agent_loc_size{1}(2);

        % Calculate Z_t and D values
        Z_t = [bus_for_DG dg_size];
        D_alpha=abs(C*Z_alpha - Z_t);
        D_beta = abs(C*Z_beta - Z_t);
        D_delta = abs(C*Z_delta - Z_t);

        % Calculate following positions
        Z_following_alpha= Z_alpha- A*D_alpha;
        Z_following_beta= Z_beta- A*D_beta;
        Z_following_delta= Z_delta- A*D_delta;
        
        % Search for nearest positions
        [Z_t_1_alpha,  Z_t_1_alpha_indices]= search(bus_sizes_matrix,Z_following_alpha);
        [Z_t_1_beta, Z_t_1_beta_indices] = search(bus_sizes_matrix,Z_following_beta);
        [Z_t_1_delta, Z_t_1_delta_indices] = search(bus_sizes_matrix,Z_following_delta);

        % Calculate final position
        Z_final_temp = (Z_t_1_alpha + Z_t_1_beta + Z_t_1_delta)/3;
        [Z_final, Z_final_indices] = search(bus_sizes_matrix,Z_final_temp);

        % GWO calculations
        i_t_1 = Z_final_indices(1);
        j_t_1 = Z_final_indices(2);
        agent_loc_size = bus_sizes_matrix(i_t_1,j_t_1);
        bus_for_DG_GWO = agent_loc_size{1}(1);
        dg_size_GWO = agent_loc_size{1}(2);

        % Update BD_temp and calculate new objective function
        BD_temp = BD;
        old_Pload = BD(bus_for_DG_GWO, 2);     
        BD_temp(bus_for_DG_GWO, 2) = old_Pload - dg_size_GWO;
        [Tl, TQl, Voltage_GWO, Vangle, Ploss_GWO, Qloss_GWO] = calculate_power_losses(LD,BD_temp,  Z, V, Sbase, num_iter);
        agent_obj_GWO = calculateObjectiveFunction(Voltage_GWO, BD_temp, LD, Ploss_GWO,Qloss_GWO);

        % DLH calculations
        del_i=abs(i_t - i_t_1);
        del_j=abs(j_t - j_t_1);

        i_neigh_gwo = randi([abs(i_t-del_i), abs(i_t+del_i)]); 
        j_neigh_gwo = randi([abs(j_t-del_j), abs(j_t+del_j)]); 

        i_neigh_pop = randi([1, length(selected_buses)]);
        j_neigh_pop = randi([1, no_of_dg_avail]);

        Z_DLH_indices = [0 0];

        % Generate random indices
        Z_DLH_indices(1) = randi([ abs(i_t-abs(i_neigh_pop- i_neigh_gwo)), abs(i_t+abs(i_neigh_pop- i_neigh_gwo))]);
        Z_DLH_indices(2) = randi([ abs(j_t-abs(j_neigh_pop- j_neigh_gwo)), abs(j_t+abs(j_neigh_pop- j_neigh_gwo))]);

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

        % Extract Z_DLH values
        Z_DLH = bus_sizes_matrix(Z_DLH_indices(1),Z_DLH_indices(2));
        bus_for_DG_DLH = Z_DLH{1}(1);
        dg_size_DLH = Z_DLH{1}(2);

        % Update BD_temp and calculate new objective function
        BD_temp = BD;
        old_Pload = BD(bus_for_DG_DLH, 2);
        BD_temp(bus_for_DG_DLH, 2) = old_Pload - dg_size_DLH;
        [Tl, TQl, Voltage_DLH, Vangle, Ploss_DLH, Qloss_DLH] = calculate_power_losses(LD,BD_temp,  Z, V, Sbase, num_iter);
        agent_obj_DLH = calculateObjectiveFunction(Voltage_DLH, BD_temp, LD, Ploss_DLH, Qloss_DLH);

        % Update agent_obj, bus_for_DG, dg_size, and location_agent based on GWO or DLH
        agent_obj=agent_obj_GWO;
        bus_for_DG= bus_for_DG_GWO;
        dg_size= dg_size_GWO;
        location_agent{cur_agent}=Z_final_indices;
        Ploss=Ploss_GWO;
        Qloss=Qloss_GWO;
        Voltage=Voltage_GWO;

        if agent_obj_GWO>agent_obj_DLH
            agent_obj=agent_obj_DLH;
            bus_for_DG= bus_for_DG_DLH;
            dg_size= dg_size_DLH;
            location_agent{cur_agent}=Z_DLH_indices;
            Ploss=Ploss_DLH;
            Qloss=Qloss_DLH;
            Voltage=Voltage_DLH;
        end

        % Constraint Check
        constraint_broken = 0; % Initialize constraint flag
        for i = 1:69 % Loop through each bus
            % Check if voltage is outside the acceptable range
            if Voltage(i) > 1.1 || Voltage(i) < 0.9
                constraint_broken = 1; % Set flag if constraint is broken
            end
        end

        % If constraint is broken, skip to the next iteration
        if constraint_broken == 1
            continue;
        end
        
        % Update agent's location
        location_agent{cur_agent} = Z_final_indices;
        
        % Update alpha, beta, delta if necessary
        if agent_obj < alpha(3)
            alpha(1) = bus_for_DG;
            alpha(2) = dg_size;
            alpha(3) = agent_obj;
            alpha(4) = cur_agent;
            % Calculate final loss based on current power losses
            complexVector = sum(Ploss) + 1i * sum(Qloss); % Combine real and imaginary parts
            final_loss = norm(complexVector); % Calculate the magnitude
            final_Voltage = Voltage; % Update final Voltage
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
% Objective function
function OBJ = calculateObjectiveFunction(Voltage, BD, LD, Ploss,Qloss)
    % Initialize variables for voltage deviation, and voltage stability index
    TVD = 0;
    VSI = 0;
    
    % Loop through each bus except the reference bus (assumed to be bus 1)
    for x = 2:69
        % Extract current voltage, real power, and imaginary power at the current bus
        curvol = Voltage(x,1);
        curpowerreal = BD(x,2);
        curpowerim = BD(x,3);
        curR = LD(x-1,4);
        curX = LD(x-1,5);
        
     
        % Accumulate voltage deviation squared
        TVD = TVD + ((curvol - Voltage(1,1))^2);

        % Accumulate voltage stability index
        VSI = VSI + curvol^4 - (4*(curpowerreal*curR + curpowerim*curX)*curvol^2) - (4*(curpowerreal*curX + curpowerim*curR));
    end
    
    % Combine real and imaginary parts of power losses to get the total power loss magnitude
    complexVector = sum(Ploss) + 1i*sum(Qloss); 
    s_loss = norm(complexVector); 
    
    % Define weights for the objective function components
    F1 = s_loss; % Total power loss magnitude
    F2 = TVD; % Voltage deviation squared
    F3 = 1/VSI; % Voltage squared integral
    R1 = 1/3; % Weight for total power loss magnitude
    R2 = 1/3; % Weight for voltage deviation squared
    R3 = 1/3; % Weight for voltage stability index
    
    % Calculate the objective function value
    OBJ = F1*R1 + F2*R2 + F3*R3;
end

function [Tl, TQl, Voltage, Vangle, Ploss, Qloss] = calculate_power_losses(LD,BD,  Z, V, Sbase, Iter)
    % Function to calculate power losses and voltages using backward-forward sweep algorithm
    % Inputs:
    %   LD: Load data matrix
    %   BD: Bus data matrix
    %   Z: Line impedance vector
    %   V: Initial voltage vector
    %   Sbase: System base power
    %   Iter: Maximum number of iterations
    
    % Convert bus data to complex load per unit values
    Sload=complex(BD(:,2),BD(:,3));
    
    % Initialize variables for load current and line current
    Iload = zeros(size(V));
    Iline = zeros(size(LD, 1), 1);
    
    % Perform backward sweep for line current calculations
    for i = 1:Iter
        Iload = conj(Sload./V);
        for j = size(LD,1):-1:1
            c = [];
            e = [];
            [c e] = find(LD(:,2:3)==LD(j,3));
            if size(c,1)==1
                Iline(LD(j,1)) = Iload(LD(j,3));
            else
                Iline(LD(j,1)) = Iload(LD(j,3)) + sum(Iline(LD(c,1))) - Iline(LD(j,1));
            end
        end
        
        % Perform forward sweep for bus-voltage calculations
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

     