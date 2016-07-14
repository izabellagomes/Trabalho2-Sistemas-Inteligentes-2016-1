%inicializacoes
clear; close all; clc
fprintf('tecle <enter> para iniciar\n');
pause
[images,labels,images_test,labels_test] = readMNIST();
fprintf('entradas e respectivas saídas desejadas foram carregadas.Tecle <enter> para continuar \n');
pause
map_dimension = 10;
[w_Kohonen, nodes_sort] = train_KohonenAlgorithmSOM(images, labels, map_dimension);
err_Kohonen = test_KohonenAlgorithSOM(w_Kohonen, nodes_sort, images_test,labels_test, map_dimension);

for l = 1:(map_dimension*map_dimension)
    subplot(map_dimension,map_dimension,l);
    axis off;
    imagesc(reshape(w_Kohonen(:,l),28,28));
    axis off;
end
hold off;

[w, sse, err] = train_backpropagation(images,labels, w_Kohonen, map_dimension, nodes_sort);
fprintf('rede treinada. Tecle <enter> para continuar\n');
pause
[sse_test, err_test] = testing(w, images_test, labels_test, w_Kohonen, map_dimension);
fprintf('rede testada. Tecle <enter> para continuar \n');
pause