function [venus cfp venus_bg cfp_bg linkage] = cell_linking(input_directory,targetimsize)

dircontents = dir(input_directory);
c = 0;
for k = 1:length(dircontents)
    if endsWith(dircontents(k).name,'_data.mat') == 1                
        c = c+1;
    end                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
end       
nummat = c;

% create a distance matrix between each cell and all cells in subsequent frame
cfp = [];
cfp_bg = [];
venus = [];
venus_bg = [];

t_cfp = [];
t_cfp_bg = [];
t_venus = [];
t_venus_bg = [];
        
        
cur_rc = [];
linkage = [];
for i = 1:nummat  
    imdata = load(fullfile(input_directory,[zerostr(5,i-1) '_data.mat']));
    if i == 1
        % reset variables

        cur_rc = [];                
        % load all cells in current image 
        for l = 1:length(imdata.var)

            [I,J] = ind2sub(targetimsize,imdata.var(l).PixelIdxList);                    
            cur_rc(l,:) = [mean(I) mean(J)]; % these are the centers of all the cells
            venus(l,i) = mean(mean(imdata.var(l).venuscell));
            venus_bg(l,i) = mean(mean(imdata.var(l).venusbg));
            cfp(l,i) =  mean(mean(imdata.var(l).cfpcell));
            cfp_bg(l,i) =  mean(mean(imdata.var(l).cfpbg));

            linkage(l,i) = l;
        end
    else
        % reset variables

        next_rc = [];

         % load all cells in next image 
        for l = 1:length(imdata.var)

            [I,J] = ind2sub(targetimsize,imdata.var(l).PixelIdxList);
            next_rc(l,:) = [mean(I) mean(J)];                    
        end
        % compute distance of each point in the cur_rc from the
        % next (ith) image           
        idx = [];
        for l = 1:size(cur_rc,1)
            dist_mat = sqrt((cur_rc(l,1) - next_rc(:,1)).^2 + (cur_rc(l,2) - next_rc(:,2)).^2);
            if numel(find(dist_mat == min(dist_mat))) == 1
                idx = find(dist_mat == min(dist_mat));
                cur_rc(l,:) = next_rc(idx,:);

                venus(l,i) = mean(mean(imdata.var(idx).venuscell));
                venus_bg(l,i) = mean(mean(imdata.var(idx).venusbg));
                cfp(l,i) =  mean(mean(imdata.var(idx).cfpcell));
                cfp_bg(l,i) =  mean(mean(imdata.var(idx).cfpbg));

                linkage(l,i) = idx;
            else
                cur_rc(l,:) = cur_rc(l,:);

                venus(l,i) = mean(mean(imdata.var(i-1).venuscell));
                venus_bg(l,i) = mean(mean(imdata.var(i-1).venusbg));
                cfp(l,i) =  mean(mean(imdata.var(i-1).cfpcell));                        
                cfp_bg(l,i) =  mean(mean(imdata.var(i-1).cfpbg));                       

                linkage(l,i) = linkage(l,i-1);
            end
        end                
    end                                                          
end