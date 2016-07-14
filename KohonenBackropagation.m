function distances = KohonenBackropagation(image, w_Kohonen, map_dimension)

    nodes_number = map_dimension * map_dimension;
    distances = sum(sqrt((w_Kohonen - repmat(image, 1, nodes_number)).^2), 1);

end

