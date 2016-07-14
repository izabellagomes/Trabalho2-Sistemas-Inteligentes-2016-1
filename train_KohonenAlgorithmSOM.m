function [w, nodes_sort] = train_KohonenAlgorithmSOM(images, labels, map_dimension)
%% images contains all of the input images of MNIST databse.
%   The size of each image is 28x28 = 784 pixels.

%% setting the inicial parameters for the map

nodes_number = map_dimension * map_dimension;

% initializing the wheights for each node in the map
w = rand(784, nodes_number);
% setting the initial learning rate
eta_inicial = 0.1;
% parameter that indicates the learning that is updated
% in every epoch and begin with the inicial value of
%eta_inicial
eta_epoch = eta_inicial;
%time constant for calculating the learning rate
tau_2 = 3000;
%setting the map index
[i, j] = ind2sub([map_dimension, map_dimension], 1:nodes_number);
%obtaining the number of images to train the map
images_number = size(images,2);
%setting the inicial value of neighbor's size
sig_inicial = map_dimension/2;
%the neighborhood for each interation
sig_epoch = sig_inicial;
% time constant for updating sigma
tau_1 = 1000/log(sig_epoch);
% classify the map neurons with the label of the input images
nodes_sort=zeros(nodes_number,1);
% number of epoch
epoch_number = 20;
% variable used to determine the period that the parcial weights
% will be shown
t_show = 5;

%% SOM Algorithm
for epoch = 1:epoch_number
    epoch
    for image = 1:images_number
        actual_image = images(:,image);
        distances = sum(sqrt((w - repmat(actual_image, 1, nodes_number)).^2), 1);
        
        %fiding the winner
        [winner, winner_index] = min(distances);
        %sorting the neurons with input images label
        nodes_sort(winner_index) = labels(image);
        %defining the discrete position of winning neuron winner
        ri = [i(winner_index), j(winner_index)];
        %calculating the distance between the winner neuron and
        %the excited neuron in the output space
        distance_squared = ([i(:), j(:)] - repmat(ri, nodes_number,1)).^2;
        %applying the neighborhood function
        % see that 1/sqrt(2*pi*sigN) is the normalization factor
        h_epoch = (1/(sqrt(2*pi)*sig_epoch)).* exp(sum(distance_squared,2)/(-2*sig_epoch));
        for actual_node = 1:nodes_number
           w(:, actual_node) = w(:, actual_node) + eta_epoch * h_epoch(actual_node) .* (actual_image - w(:, actual_node));
        end
    end
    %updating the learning rate
    eta_epoch = eta_inicial * exp(-epoch/tau_2);
    %updating the neighborhood
    sig_epoch = sig_inicial * exp(-epoch / tau_1);

    %show the weights every t_show epoch
    %//show the weights every 100 epoch
    if mod(epoch,t_show) == 0
        figure;
        axis off;
        hold on;
        nodes_sort
        for l = 1:nodes_number
            subplot(map_dimension,map_dimension,l);
            axis off;
            imagesc(reshape(w(:,l),28,28));
            axis off;
        end
        hold off;
    end
end

end