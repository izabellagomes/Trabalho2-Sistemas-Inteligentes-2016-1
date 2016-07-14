function [w, sse, err] = train_backpropagation(images,labels, w_Kohonen, map_dimension, nodes_sort)
                        %I: entradas vetor de 60000 posicoes com matrizes de 28x28 em cada
                        %labels: valores de saida desejados correspondentes a cada entrada

func_ativacao = 'logistica';
nep = 100; % Número de épocas
eta = 0.05; % Taxa de aprendizado
n_camadas = 2;
n_saidas = 10;
n_entradas = map_dimension * map_dimension;

%[m0, n0] = size(images);
%images = [images; ones(1,n0)]; %imagens +bias

formato_camadas = [100 n_saidas; n_entradas+1 101]; %1a linha: qdt neuronios, 2a linha qtd entradas + bias

for i = 1:n_camadas
    w{i} = rand(formato_camadas(:,i)')*0.2-0.1; % peso de cada camada
                                                % + peso do bias
end



sse=zeros(n_saidas,nep); %Soma de erros quadráticos
err=zeros(nep,1); %Total de padrões errados



%images=images';
images=double(images);
if strcmp(func_ativacao, 'logistica')
    
    d = zeros(n_saidas, 60000); %valores desejados de saida
    for i = 1:n_saidas
        for j = 1:60000
            d(i,j) = 0.1; % valores para as saídas que
                          %nao correspondem aa entrada
        end
    end
    for i = 1:60000
       if labels(i) == 0
            d(n_saidas, i) = 0.8;   % valores para as saidas que
                                    % correspondem aa entrada
        else
            d(labels(i), i) = 0.8;
       end
    end
    
    for i = 1:nep
        fprintf('epoca: ');
        i
        fprintf('\n');
        for j=1:60000 % cada imagem
          distances = KohonenBackropagation(images(:, j), w_Kohonen, map_dimension);
          distances = [distances/norm(distances,Inf), 1];
          %Propagação
          s1 = w{1} * distances';
          y{1} = 1.0./(1+exp(-s1));

          for k = 2:n_camadas
              s = w{k}*[y{k-1};1]; % pesos * (entradas + bias)
              y{k}=1./(1+exp(-s));
          
          end
          
          sse(:, i) = sse(:, i)+(d(:, j)-y{n_camadas}).^2; % Calcula erro
          
          [maxY, index] = max(y{n_camadas});
          if(index == 10 && labels(j) ~= 0)
              err(i) = err(i) + 1;
          else
              if(index ~= labels(j))
                  err(i) = err(i) + 1;
              end
          end
                         %Retropropagação
          e{n_camadas}=(d(:,j)-y{n_camadas}).*y{n_camadas}.*(1-y{n_camadas}); %Erro na saída
          dw{n_camadas}=e{n_camadas}*transpose([y{n_camadas - 1};1]);

          for k = 1:n_camadas-2
            [n_linhas, n_colunas] = size(w{n_camadas - k+1});
            e{n_camadas-k} = transpose(w{n_camadas - k+1}(:,1:(n_colunas - 1)))*e{n_camadas-k+1}.*y{n_camadas - k}.*(1-y{n_camadas - k});
            dw{n_camadas - k} = e{n_camadas-k}*transpose([y{n_camadas - k - 1};1]);
          end

          [n_linhas, n_colunas] = size(w{2});
          e{1} = transpose(w{2}(:,1:n_colunas - 1))*e{2}.*y{1}.*(1-y{1}); % Propagação do erro para 1a camada
          dw{1} = e{1} * distances;

          for k = 1:n_camadas
            w{k} = w{k}+eta*dw{k}; % Atualização
          end
        end
    end
else
    if strcmp(func_ativacao, 'tanh')
    
    for i = 1:n_saidas
        for j = 1:60000
            d(i,j) = -0.5;
        end
    end
    for i = 1:60000
       if labels(i) == 0
            d(n_saidas, i) = 0.5;
        else
            d(labels(i), i) = 0.5;
       end
    end
    
    for i=1:nep
        fprintf('epoca: ');
        i
        fprintf('\n');
        for j=1:60000 %
          %Propagação
          distances = [KohonenBackropagation(images(:, j), w_Kohonen, map_dimension), 1];
          s1 = w{1}*distances;
          y{1} = tanh(s1);

          for k = 2:n_camadas
              s = w{k}*[y{k-1};1];
              y{k}=tanh(s); 
          end
          sse(:, i)=sse(:, i)+(d(:, j)-y{n_camadas}).^2; % Calcula erro
          
          [maxY, index] = max(y{n_camadas});
          
          if(index == 10 && labels(j) ~= 0)
              err(i) = err(i) + 1;
          else
              if(index ~= labels(j))
                  err(i) = err(i) + 1;
              end
          end
                         %Retropropagação
          e{n_camadas}=(d(:,j)-y{n_camadas}).*(1 - y{n_camadas}).*(1+y{n_camadas}); %Erro na saída
          dw{n_camadas}=e{n_camadas}*transpose([y{n_camadas - 1};1]);

          for k = 1:n_camadas - 2
            [n_linhas, n_colunas] = size(w{n_camadas - k+1});
            e{n_camadas-k} = transpose(w{n_camadas - k+1}(:,1:(n_colunas - 1))).*e{n_camadas-k+1}.*(1 - y{n_camadas - k}).*(1 + y{n_camadas - k});
            dw{n_camadas - k} = e{n_camadas-k}*transpose([y{n_camadas - k - 1};1]);
          end

          [n_linhas, n_colunas] = size(w{2});
          e{1} = transpose(w{2}(:,1:n_colunas - 1))*e{2}.*(1 - y{1}).*(1 + y{1}); % Propagação do erro para 1a camada
          dw{1}=e{1}*transpose(distances);

          for k = 1:n_camadas
            w{k}=w{k}+eta*dw{k}; % Atualização
          end
        end
    end
    end
   
end
end