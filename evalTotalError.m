function [cell_intensity_truth_total,cell_intensity_DL_total] =  evalTotalError(imds,pxdsTruth,pxdsResults,n_images,targetimsize)


cell_intensity_truth_total = [];
cell_intensity_DL_total = [];

for i = 1:n_images
    disp([num2str(i) ' / ' num2str(n_images)]) 
    % load real image
    real_cell = double(imread(imds.Files{i}));

    % load truth image
    truth_labels = imread(pxdsTruth.Files{i});

    % load DL labeled image
    DL_labels = imread(pxdsResults.Files{i});

    % compare cell body+nucleus for truth to DL labeled
    cytoplasm_truth = truth_labels == 2;
    nuclei_truth = truth_labels == 3;
    total_truth = cytoplasm_truth+nuclei_truth;
    truth_cell = real_cell.*total_truth;
    bw_truth = bwlabel(total_truth);
    
    [filtered_image filtered_nuclei] = filter_network_output(DL_labels+1);
    filtered_image = filtered_image > 1 == 1;
    filtered_image = bwlabel(filtered_image);
    
    % get pixels
    regions_truth = regionprops(bw_truth,'PixelIdxList','PixelList');
    regions_DL = regionprops(filtered_image,'PixelIdxList','PixelList');
                    
    
    % reset variables
    cell_intensity_DL = [];
    cell_intensity_truth = [];
    cur_rc = [];                
    % load all cells in truth image 
    for l = 1:length(regions_DL)
%                     cur_boundingboxes(l,:) = imdata.var(l).BoundingBox;
        [I,J] = ind2sub(targetimsize,regions_DL(l).PixelIdxList);                    
        cur_rc(l,:) = [mean(I) mean(J)]; % these are the centers of all the cells
        cell_intensity_DL(l) = mean(mean(real_cell(regions_DL(l).PixelIdxList)));

%         linkage(l) = l;
    end
    % reset variables

    next_rc = [];
     % load all cells in next image 
    for l = 1:length(regions_truth)
%                     boundingboxes(l,:) = imdata.var(l).BoundingBox;
        [I,J] = ind2sub(targetimsize,regions_truth(l).PixelIdxList);
        next_rc(l,:) = [mean(I) mean(J)];                    
    end
    % compute distance of each point in the cur_rc from the
    % next image (here, its the DL image)
    idx = [];
    for l = 1:size(cur_rc,1)
        dist_mat = sqrt((cur_rc(l,1) - next_rc(:,1)).^2 + (cur_rc(l,2) - next_rc(:,2)).^2);
            idx = min(find(dist_mat == min(dist_mat)));
            cur_rc(l,:) = next_rc(idx,:);
            cell_intensity_truth(l) = mean(mean(real_cell(regions_truth(idx).PixelIdxList)));      

    end                
    cell_intensity_truth_total = [cell_intensity_truth_total cell_intensity_truth];
    cell_intensity_DL_total = [cell_intensity_DL_total cell_intensity_DL];
end


% filter out zeros
idx = find(cell_intensity_truth_total == 0);
cell_intensity_truth_total(idx) = [];
cell_intensity_DL_total(idx) = [];
idx = find(cell_intensity_DL_total == 0);
cell_intensity_truth_total(idx) = [];
cell_intensity_DL_total(idx) = [];