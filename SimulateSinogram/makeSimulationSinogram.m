% Simple Simulation of x-ray phantom

format long

c_o = 0.8; % Liquidus composition of element A.
k = .5; %equilibrium partition coefficient
speed = .1; %growth speed: lower means slower (10, approx solns to 250), high means faster (solns to inf.)
angle_step = 5;
nuc_time = 100;
tot_time = 1000;

% Simulation parameters
grid_size = 500;
rMax = .1*grid_size;
circ_radii = grid_size/2.5;
circ_center = [grid_size/1.8 grid_size/1.8];
circleMask = makeCircle(ones(grid_size), circ_radii, circ_center(1), circ_center(2));

simpleGrid = ones(grid_size).*circleMask;
liq_size = sum(simpleGrid, 'all');

Elem_A = simpleGrid*c_o;
Elem_B = simpleGrid*(1-c_o);

mass_A_total = sum(Elem_A, 'all');

% Growth descbited by analytically derived solution from:
% Lorenz Ratke and P. W Voorhees. Growth and coarsening: Ostwald ripening in material processing. OCLC:751525888. Berlin; London: Springer, 2011. ISBN: 978-3-642-07644-2.
%tau = 100;

% format long
% eqn_int = @(r) r./(1-r.^3);
% upper_bound = 0.000001:0.000001:0.999999;
% solution = zeros(size(upper_bound));
% index = 1;
% for ub = upper_bound
%     disp(ub)
%     solution(index) = integral(eqn_int, 0, ub, 'RelTol',0,'AbsTol',1e-12);
%     index=index+1;
% end
% figure(4), plot(round(solution*tau), upper_bound)
% box on, xlabel('time'), ylabel('Fraction solid'), set(gcf, 'color','w')
load solution.mat 
load upper_bound.mat
load tracking_wp.mat

r_evolution = round(solution*grid_size/speed);
r_increment = 0;

disp('Begin simulation...' )
disp(['Grid size: ' num2str(grid_size)])
disp(['Nucleation time: ' num2str(nuc_time) '/' num2str(tot_time)])
index_increment=0;

attn_A = 1000; %2800; %for 27keV
attn_B = 500; %780; %for 27keV
I_0 = 100;
sinogram = zeros(grid_size, tot_time);
angle = 0;

x = 0.6*grid_size;
y = 0.6*grid_size;

file_name_out = ['new_sinogram_spd' num2str(speed) '_k' num2str(k) '_co' num2str(c_o) '.h5']

amount_off_border = 2/3;


tic
for time = 1:tot_time
    %disp(time)
    if time < nuc_time
        c_L = c_o;
        Solid_pixels = zeros(grid_size);
        Liquid_pixels = zeros(grid_size);
    else
        
      index_list = find(r_evolution == r_increment);
      if isempty(index_list)
            rFrac = .9999;
      else
            rFrac = upper_bound(index_list(1));
      end
      
      index_increment = index_increment +1;
      rad = rFrac * rMax;
      
      Solid_pixels = makeCircle(ones(grid_size),rad, x, y);
      Liquid_pixels =  ~Solid_pixels & circleMask;
      
      r_increment = r_increment+1;
      
      c_L = c_L_prev;
    end 
      
    c_S = k*c_L; %k = C_s/c_L

    %Update solid composition
    Elem_A(Solid_pixels == 1) = c_S;
    Elem_B(Solid_pixels == 1) = 1-c_S;

    total_A_solid = sum(Solid_pixels, 'all')*c_S;
    
    %Back fill liquid with the amount of mass to distribute
    total_A_liq = mass_A_total - total_A_solid;
    if sum(Liquid_pixels, 'all') == 0
        c_L = c_o;
    else
        c_L = total_A_liq/sum(Liquid_pixels, 'all');
    end
    c_L_prev = c_L;

    Elem_A(Liquid_pixels == 1) = c_L;
    Elem_B(Liquid_pixels == 1) = 1-c_L;

    Elem_A = Elem_A.*circleMask;
    Elem_B = Elem_B.*circleMask;

    sliceA = imrotate(Elem_A, angle, 'crop');
    sliceB = imrotate(Elem_B, angle, 'crop');
    sinogram(:,time) = I_0*exp(-sum(sliceA,1)/attn_A + -sum(sliceB, 1)/attn_B);
    angle = angle + angle_step;
end
toc

data_export = sinogram;
h5create(file_name_out,'/data',size(data_export))
h5write(file_name_out,'/data',data_export)


function circ = makeCircle(I, radii, xc, yc)
% 
    % make circle mask
    [xDim,yDim] = size(I);
    %center = [xDim*.7 yDim*.7];
    %xc = center(1);
    %yc = center(2);
    %%pos = [xDim/scalar yDim/scalar xDim-xDim/scalar/2 yDim-yDim/scalar/2];
    %  circ = rectangle('Position',pos,'Curvature',[1 1], 'EdgeColor', 'None');
    [xx,yy] = meshgrid(1:yDim,1:xDim);
    mask = false(xDim,yDim);
    circ = mask | hypot(xx - xc, yy - yc) <= radii;
end
