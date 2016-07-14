function err_Kohonen = test_KohonenAlgorithSOM( w, nodes_sort, images_test,labels_test, map_dimension)
%% setting the inicial parameters for the map test

nodes_number = map_dimension * map_dimension;
%obtaining the number of images to test the map
images_number = size(images_test,2);
% number of epoch
epoch_number = 20;
%number of erros
err_Kohonen = zeros(epoch_number,1);

%%
for epoch = 1:epoch_number
    epoch
    for image = 1:images_number
        actual_image = images_test(:,image);
        distances = sum(sqrt((w - repmat(actual_image, 1, nodes_number)).^2), 1);
        %fiding the winner
        [winner, winner_index] = min(distances);
        if nodes_sort(winner_index) ~= labels_test(image)
            err_Kohonen(epoch) = err_Kohonen(epoch) + 1;
        end
    end
end

end

