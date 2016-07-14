function [sse_test, err_test] = testing(w, images_test, labels_test, w_Kohonen, map_dimension)

func_ativacao = 'logistica';
nep = 100; % Número de épocas
n_camadas = 2;
n_saidas = 10;

sse_test=zeros(n_saidas,nep); %Soma de err_testes quadráticos
err_test=zeros(nep,1); %Total de padrões err_testados

images_test = double(images_test);
    for i=1:nep
        fprintf('epoca: ');
        i
        fprintf('\n');
        for j=1:10000 % 
          distances = KohonenBackropagation(images_test(:, j), w_Kohonen, map_dimension);
          distances = [distances/norm(distances,Inf), 1];
          %Propagação
          s1 = w{1} * distances';
          y{1} = 1.0./(1+exp(-s1));

          for k = 2:n_camadas
              s = w{k}*[y{k-1};1]; % pesos * (entradas + bias)
              y{k}=1./(1+exp(-s));
          
          end
          [maxY, index] = max(y{n_camadas});
          if(index == 10 && labels_test(j) ~= 0)
              err_test(i) = err_test(i) + 1;
          else
              if(index ~= labels_test(j))
                  err_test(i) = err_test(i) + 1;
              end
          end
        end
    end
end

