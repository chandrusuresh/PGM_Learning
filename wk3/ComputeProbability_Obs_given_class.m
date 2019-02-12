function prob = ComputeProbability_Obs_given_class(P,G,dataset,class)
j = class;
prob = 0;
for i = 1:size(dataset,1)
    for k = 1:size(G,1)
        if G(k,1) == 0
            prob = prob + lognormpdf(dataset(i,k,1),P.clg(k).mu_y(j),P.clg(k).sigma_y(j)) + ...
                      lognormpdf(dataset(i,k,2),P.clg(k).mu_x(j),P.clg(k).sigma_x(j)) + ...
                      lognormpdf(dataset(i,k,3),P.clg(k).mu_angle(j),P.clg(k).sigma_angle(j));
        else
            parent_val = dataset(i,G(k,2),:);
            data_val   = dataset(i,k,:);
            mu_y = sum(P.clg(k).theta(j,1:4).*[1,parent_val(:)']);
            mu_x = sum(P.clg(k).theta(j,5:8).*[1,parent_val(:)']);
            mu_angle = sum(P.clg(k).theta(j,9:12).*[1,parent_val(:)']);
            prob = prob + lognormpdf(data_val(1,1,1),mu_y,P.clg(k).sigma_y(j)) + ...
                      lognormpdf(data_val(1,1,2),mu_x,P.clg(k).sigma_x(j)) + ...
                      lognormpdf(data_val(1,1,3),mu_angle,P.clg(k).sigma_angle(j));
        end
    end
end