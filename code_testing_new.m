% code for STAR methods
%% part 1- creating deep learning algorithm
clear
addpath('C:\Program Files\MATLAB\R2018b\examples\images\main')
basedir = 'C:\Users\Dave\Box Sync\LPR_Protocol\code_testing\images';
ground_truth = 'C:\Users\Dave\Box Sync\LPR_Protocol\code_testing\images\groundtruth';

classNames = ["background","border","cell","nuclei"];

num_images = 434;
frac_training = 0.7;
[training, validation, testing] = randomize_images(num_images,frac_training);

image_directory = basedir;
divide_images(image_directory,ground_truth,training,validation,testing);

imds = imageDatastore(fullfile(basedir,'training'),'FileExtensions',{'.png'});
classNames = ["background","border","cell","nuclei"];
pixelLabelIds = [0 1 2 3];
pxds = pixelLabelDatastore(fullfile(basedir,'training_label'),classNames,pixelLabelIds);
imds_validation =  imageDatastore(fullfile(basedir,'validation'),'FileExtensions',{'.png'});
pxds_validation =  pixelLabelDatastore(fullfile(basedir,'validation_label'),classNames,pixelLabelIds);
pximds_validation = pixelLabelImageDatastore(imds_validation,pxds_validation);

augmenter = imageDataAugmenter('RandRotation',[-180 180],...
    'RandXReflection',true,...
    'RandYReflection',true,...
    'RandXShear',[0 10],...
    'RandYShear',[0 10]);

patchds = randomPatchExtractionDatastore(imds,pxds,128,'PatchesPerImage',256,'DataAugmentation',augmenter);
patchds_validation = randomPatchExtractionDatastore(imds_validation,pxds_validation,128,'PatchesPerImage',256);

minibatch = preview(patchds);
montage(minibatch.InputImage,'DisplayRange',[0 255])


tbl = countEachLabel(pxds);
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency;

numFilters = 64; % explanation
filterSize = 3; % explanation
numClasses = length(pixelLabelIds);
layers = [
       imageInputLayer([128 128 1],'Name','ImageInputLayer')
    
       convolution2dLayer(filterSize,numFilters,...
'Padding','same','NumChannels',1,...
        'Name','Encoder-Stage-1-Conv-1')
       reluLayer(...
        'Name','Encoder-Stage-1-ReLU-1')
       convolution2dLayer(filterSize,numFilters,...
'Padding','same','NumChannels',numFilters,...
        'Name','Encoder-Stage-1-Conv-2')
       reluLayer(...
        'Name','Encoder-Stage-1-ReLU-2')    
       maxPooling2dLayer(2,'Stride',2,...
        'Name', 'Encoder-Stage-1-MaxPool' )
    
       convolution2dLayer(filterSize,numFilters*2,...
'Padding','same','NumChannels',numFilters,...
        'Name','Encoder-Stage-2-Conv-1')
       reluLayer(...
        'Name','Encoder-Stage-2-ReLU-1')
       convolution2dLayer(filterSize,numFilters*2, ...
'Padding','same','NumChannels',numFilters*2,...
        'Name','Encoder-Stage-2-Conv-2')
       reluLayer(...
        'Name','Encoder-Stage-2-ReLU-2')    
    maxPooling2dLayer(2,'Stride',2,...
        'Name','Encoder-Stage-2-MaxPool')
    
      convolution2dLayer(filterSize,numFilters*4, ...
'Padding','same','NumChannels',numFilters*2,...
        'Name','Encoder-Stage-3-Conv-1')
      reluLayer(...
        'Name','Encoder-Stage-3-ReLU-1')
    convolution2dLayer(filterSize,numFilters*4, ...
'Padding','same','NumChannels',numFilters*4,...
        'Name','Encoder-Stage-3-Conv-2')
    reluLayer(...
        'Name','Encoder-Stage-3-ReLU-2')    
    dropoutLayer(...
        'Name','Encoder-Stage-3-DropOut')
    maxPooling2dLayer(2,'Stride',2,...
        'Name','Encoder-Stage-3-MaxPool')
    
    convolution2dLayer(filterSize,numFilters*8, ...
'Padding','same','NumChannels',numFilters*4,...
        'Name','Bridge-Conv-1' )
    reluLayer(...
        'Name','Bridge-ReLU-1')    
    
    convolution2dLayer(filterSize,numFilters*8, ...
'Padding','same','NumChannels',numFilters*8,...
        'Name','Bridge-Conv-2')
    reluLayer(...
        'Name','Bridge-ReLU-2')    
    dropoutLayer(...
        'Name','Bridge-DropOut')
    transposedConv2dLayer(filterSize-1,numFilters*4, ...
'NumChannels',numFilters*8,'Stride',[2 2],...
        'Name','Decoder-Stage-1-UpConv')
    reluLayer(...
        'Name','Decoder-Stage-1-UpReLU' )    
    depthConcatenationLayer(2,...
        'Name','Decoder-Stage-1-DepthConcatenation')
    convolution2dLayer(filterSize,numFilters*4, ...
'Padding','same','NumChannels',numFilters*8,...
        'Name','Decoder-Stage-1-Conv-1')
    reluLayer(...
        'Name','Decoder-Stage-1-ReLU-1')    
    convolution2dLayer(filterSize,numFilters*4, ...
'Padding','same','NumChannels',numFilters*4,...
        'Name','Decoder-Stage-1-Conv-2')    
    reluLayer(...
        'Name','Decoder-Stage-1-ReLU-2')     
    transposedConv2dLayer(filterSize-1,numFilters*2, ...
'NumChannels',numFilters*4,'Stride',[2 2],...
        'Name','Decoder-Stage-2-UpConv')
    reluLayer(...
        'Name','Decoder-Stage-2-UpReLU' )    
    depthConcatenationLayer(2,...
        'Name','Decoder-Stage-2-DepthConcatenation')        
        
    convolution2dLayer(filterSize,numFilters*2, ...
'Padding','same','NumChannels',numFilters*4,...
        'Name','Decoder-Stage-2-Conv-1')
    reluLayer(...
        'Name','Decoder-Stage-2-ReLU-1')    
    convolution2dLayer(filterSize,numFilters*2, ...
'Padding','same','NumChannels',numFilters*2,...
        'Name','Decoder-Stage-2-Conv-2')    
    reluLayer(...
        'Name','Decoder-Stage-2-ReLU-2')     
        
    transposedConv2dLayer(filterSize-1,numFilters*1, ...
'NumChannels',numFilters*2,'Stride',[2 2],...
        'Name','Decoder-Stage-3-UpConv')
    reluLayer(...
        'Name','Decoder-Stage-3-UpReLU')             
    depthConcatenationLayer(2,...
        'Name','Decoder-Stage-3-DepthConcatenation')                
        
    convolution2dLayer(filterSize,numFilters*1, ...
'Padding','same','NumChannels',numFilters*2,...
        'Name','Decoder-Stage-3-Conv-1')
    reluLayer(...
        'Name','Decoder-Stage-3-ReLU-1')    
    convolution2dLayer(filterSize,numFilters*1, ...
'Padding','same','NumChannels',numFilters*1,...
        'Name','Decoder-Stage-3-Conv-2')    
    reluLayer(...
        'Name','Decoder-Stage-3-ReLU-2')     
        
convolution2dLayer(1,numClasses,'Padding',0, ...
'NumChannels',numFilters*1,'Name','Final-ConvolutionLayer')        
        
    softmaxLayer('Name','Softmax-Layer')
  pixelClassificationLayer('Classes',tbl.Name, ...
'ClassWeights',classWeights,...
        'Name','Segmentation-Layer')
    ];

lgraph2 = layerGraph(layers);
lgraph2 = connectLayers(lgraph2,'Encoder-Stage-1-ReLU-2','Decoder-Stage-3-DepthConcatenation/in2');
lgraph2 = connectLayers(lgraph2,...
'Encoder-Stage-2-ReLU-2' ,'Decoder-Stage-2-DepthConcatenation/in2');
lgraph2 = connectLayers(lgraph2,...
'Encoder-Stage-3-ReLU-2','Decoder-Stage-1-DepthConcatenation/in2');

plot(lgraph2)


%% pre-processing images
pixel_size = 11e-6; 
magnification = 10;
cellsize = 11e-6;
tophatw = 3*round(cellsize/(pixel_size/magnification));
h = fspecial('disk',tophatw);
thresh = 0;
targetimsize = [1200 1200];
targetbitsize = 8;

venusimage = 'venus.tif';
cfpimage = 'cfp.tif';

[venus_orig cfp_orig thresh_venus thresh_cfp]  = resizetophatim(venusimage,cfpimage,targetimsize,h,thresh,targetbitsize);

segnet = load('network.mat');
output = semanticseg(thresh_venus,segnet.net,'ExecutionEnvironment','gpu');


[filtered_image filtered_nuclei] = filter_network_output(output);

bw_filtered_image = bwlabel(filtered_image);
bw_filtered_nuclei = bwlabel(filtered_nuclei);

figure(6)
output_im = double(output);
bw_im = double(bw_filtered_image);
venus_im = imread(venusimage);
J = imadjust(venus_im,stretchlim(venus_im),[]);
[X2]= output_im/max(output_im(:));
[X3]= bw_im/max(bw_im(:));
map = jet;
subplot(1,3,1), imshow(J), title('Contrast adjusted Venus image')
subplot(1,3,2), imshow(X2,'Colormap',map), title('Semantic segmentation')
subplot(1,3,3), imshow(X3,'Colormap',map), title('Additional filtering')

expand = 3;

venusresize = double(imresize(imread(venusimage),targetimsize));
cfpresize = double(imresize(imread(cfpimage),targetimsize));

imdata = cell_decomposition(bw_filtered_image, bw_filtered_nuclei,venusresize,cfpresize,expand);

%% loop multiple images

% place all variables that don't change outside the loop. Recommended that
% you use GPUs to do semanticseg first, then use parallel cpus for the
% rest. 

% make output directory
output_directory = 'semantic_output';
mkdir(output_directory);

parpool('local',4); % start parallel pool
segnet = load('network.mat');

CFPc = 'CFP-427-4_6_000';
FRETc = 'FRET-427-542-6_000';
num_images = 452;
parfor i = 1:num_images
    disp(['[' num2str(i) ']/[' num2str(num_images) ']'])
    cfpimage = ['img_' zerostr(9,i-1) '_' CFPc '.tif'];
    venusimage = ['img_' zerostr(9,i-1) '_' FRETc '.tif'];

    [venus_orig cfp_orig thresh_venus thresh_cfp] = resizetophatim(...
        fullfile('demo_images',venusimage),fullfile('demo_images',cfpimage),targetimsize,h,thresh,targetbitsize);

    
    output = semanticseg(thresh_venus,segnet.net,'ExecutionEnvironment','gpu');
    
    parsave(fullfile(output_directory,[zerostr(5,i-1) '.mat']), output);    
end
%% now that the segmentation is done, loop the post-processing using parallel processing
delete(gcp('nocreate'))

parpool('local',8);

% make output directory
output_directory = 'singlecell_output';
mkdir(output_directory);

input_directory = 'semantic_output';

expand = 3; % target cell expansion

parfor i = 1:num_images
    disp(['[' num2str(i) ']/[' num2str(num_images) ']'])
    input_image = load(fullfile(input_directory,[zerostr(5,i-1) '.mat']));
    [filtered_image filtered_nuclei] = filter_network_output(input_image.var);

    bw_filtered_image = bwlabel(filtered_image);
    bw_filtered_nuclei = bwlabel(filtered_nuclei);

    cfpimage = ['img_' zerostr(9,i-1) '_' CFPc '.tif'];
    venusimage = ['img_' zerostr(9,i-1) '_' FRETc '.tif'];    
    
    venusresize = double(imresize(imread(fullfile('demo_images',venusimage)),targetimsize));
    cfpresize = double(imresize(imread(fullfile('demo_images',cfpimage)),targetimsize));

    imdata = cell_decomposition(bw_filtered_image, bw_filtered_nuclei,venusresize,cfpresize,expand);
    parsave(fullfile(output_directory,[zerostr(5,i-1) '_data.mat']),imdata);
end

%% link cells
input_directory = 'singlecell_output';
[venus cfp venus_bg cfp_bg linkage] = cell_linking(input_directory,targetimsize);

% save output
output_directory = 'singlecell_links';
mkdir(output_directory);
save(fullfile(output_directory,'linkvenus.mat'),'venus');
save(fullfile(output_directory,'linkcfp.mat'),'cfp');
save(fullfile(output_directory,'linkvenusbg.mat'),'venus_bg');
save(fullfile(output_directory,'linkcfpbg.mat'),'cfp_bg');
save(fullfile(output_directory,'linkage.mat'),'linkage');
        
%% analyze data
load(fullfile(output_directory,'linkcfp.mat'));
load(fullfile(output_directory,'linkvenus.mat'));        
load(fullfile(output_directory,'linkcfpbg.mat'));
load(fullfile(output_directory,'linkvenusbg.mat'));        

fret = cfp./venus;
fret_bg = (cfp-cfp_bg)./(venus - venus_bg);

imdata = load(fullfile('singlecell_output',[zerostr(5,0) '_data.mat']));
        
dist = 20;
solution_changes = [150 300];
clear slope gof_r2 cellsize cells
for i = 1:size(fret,1)
    mfret = medfilt1(fret(i,:),5);
    % find minimum point for each solution change
    for h = 1:length(solution_changes)
        if h ~= length(solution_changes)
            vec = mfret(solution_changes(h):solution_changes(h+1));
        else
            vec = mfret(solution_changes(h):length(mfret));
        end
        idx = find(vec == min(vec));
        if h == length(solution_changes)
            if length(mfret)-(solution_changes(h)+min(idx))  < dist
                idx = 1;
            end
        end
       x = solution_changes(h)+min(idx)-1;
        [fitobj gof] = fit((x:x+dist)',mfret(x:x+dist)','poly1');
        slope(i,h) = fitobj.p1;
        gof_r2(i,h) = gof.rsquare;
    end                        
end               

% filter and reshape 
filter = 0.9;
slope_n(1).slope = [];
for m = 1:length(slope(1,:))
    c = 1; % counter
    for i = 1:size(slope,1)
        if gof_r2(i,m) > filter
            slope_n(m).slope(c) = slope(i,m);
            slope_n(m).r2(c) = gof_r2(i,m);     
            slope_n(m).index(c) = i;
            c = c+1;
        end
    end            
end        

save results.mat slope_n

plot(fret(slope_n(2).index(50),1:450),'k.')
text(150,0.6075,'\leftarrow Glucose injection')
text(300,0.6075,'\leftarrow pCMBA injection')
xlabel('Frame')
ylabel('1/FRET')

glucose = slope_n(1).slope/0.01221;
lactate = slope_n(2).slope/0.01221;



%% test network accuracy
output_directory = 'network_test';
mkdir(output_directory)

imds = imageDatastore(fullfile(basedir, 'testing'));
pxdsTruth = pixelLabelDatastore(fullfile(basedir, 'testing_label'),classNames,pixelLabelIds);
for i = 1:65
    [C,scores] = semanticseg(imread(imds.Files{i}),segnet.net,'outputtype','uint8');
    imwrite(C-1,fullfile(output_directory,[zerostr(3,i) '.png']))
end
pxdsResults = pixelLabelDatastore(output_directory,classNames,pixelLabelIds);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);
save metrics.mat metrics
%%
cm = metrics.ConfusionMatrix;
TP = cm{1,1} + cm{2,2} + cm{3,3} + cm{4,4};
TN = (cm{2,2} + cm{3,3} + cm{4,4})/3 + (cm{1,1} + cm{3,3} + cm{4,4})/3  + (cm{1,1} + cm{2,2} + cm{4,4})/3 + (cm{2,2} + cm{3,3} + cm{4,4})/3;
FP = (cm{1,2}+cm{1,3}+cm{1,4})/3 + (cm{2,1}+cm{2,3}+cm{2,4})/3 + (cm{3,1}+cm{3,2}+cm{2,4})/3 +(cm{4,1}+cm{4,2}+cm{4,3})/3;
FN = (cm{2,1}+cm{3,1}+cm{4,1})/3 +(cm{1,2}+cm{3,2}+cm{4,2})/3 +(cm{1,3}+cm{2,3}+cm{4,3})/3+(cm{1,4}+cm{2,4}+cm{3,4})/3;
MCC = ((TP*TN) - (FP*FN))/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));

%% std deviation
mkdir('temp');
mkdir('temp_truth');
for i = 1:65
    [C,scores] = semanticseg(imread(imds.Files{i}),segnet.net,'outputtype','uint8');
    imwrite(C-1,fullfile('temp',['temp.png']))
    copyfile(fullfile(basedir,'testing_label',[zerostr(4,i) '.png']),...
        fullfile('temp_truth',['temp.png']));
    pxdsTruthTemp = pixelLabelDatastore(fullfile('temp_truth'),classNames,pixelLabelIds);
    pxdsResults = pixelLabelDatastore(fullfile('temp'),classNames,pixelLabelIds);
    metrics_temp = evaluateSemanticSegmentation(pxdsResults,pxdsTruthTemp);
    metrics_sd(i).metrics = metrics_temp;
end

save metrics_sd.mat metrics_sd
%%
for i = 1:65
    global_accuracy(i) = metrics_sd(i).metrics.DataSetMetrics{1,1};
    background_accuracy(i) = metrics_sd(i).metrics.ClassMetrics{1,1};
    background_BFscore(i) = metrics_sd(i).metrics.ClassMetrics{1,3};
    border_accuracy(i) = metrics_sd(i).metrics.ClassMetrics{2,1};
    border_BFscore(i) = metrics_sd(i).metrics.ClassMetrics{2,3};
    cell_accuracy(i) = metrics_sd(i).metrics.ClassMetrics{3,1};
    cell_BFscore(i) = metrics_sd(i).metrics.ClassMetrics{3,3};
    nuclei_accuracy(i) = metrics_sd(i).metrics.ClassMetrics{4,1};
    nuclei_BFscore(i) = metrics_sd(i).metrics.ClassMetrics{4,3};
end
std_global_accuracy = std(global_accuracy);
std_background_accuracy = std(background_accuracy);
std_border_accuracy = std(border_accuracy);
std_cell_accuracy = std(cell_accuracy);
std_nuclei_accuracy = std(nuclei_accuracy);
std_background_BFscore = std(background_BFscore);
std_border_BFscore = std(border_BFscore);
std_cell_BFscore = std(cell_BFscore);
std_nuclei_BFscore = std(nuclei_BFscore);

%%
% precision is TP / TP+FP

precision_bg = cm{1,1}/(cm{1,1} + cm{1,2} + cm{1,3} + cm{1,4})
precision_border = cm{2,2}/(cm{2,1} + cm{2,2} + cm{2,3} + cm{2,4})
precision_cell = cm{3,3}/(cm{3,1} + cm{3,2} + cm{3,3} + cm{3,4})
precision_nuclei = cm{4,4}/(cm{4,1} + cm{4,2} + cm{4,3} + cm{4,4})

% recall is TP / TP + FN

recall_bg = cm{1,1}/(cm{1,1}+cm{2,1}+cm{3,1}+cm{4,1})
recall_border = cm{2,2}/(cm{1,2}+cm{2,2}+cm{3,2}+cm{4,2})
recall_cell = cm{3,3}/(cm{1,3}+cm{2,3}+cm{3,3}+cm{4,3})
recall_nuclei = cm{4,4}/(cm{1,4}+cm{2,4}+cm{3,4}+cm{4,4})


%% Dice score
% use network on ground truth to calculate dice score
for i = 1:65
    test_image = double(imread(fullfile(basedir,'testing',[zerostr(4,i) '.png'])));
    truth_image = double(imread(fullfile(basedir,'testing_label',[zerostr(4,i) '.png'])));
    test_image_class_0 = test_image == 0;
    truth_image_class_0 = truth_image == 0;
    test_image_class_1 = test_image == 1;
    truth_image_class_1 = truth_image == 1;
    test_image_class_2 = test_image == 2;
    truth_image_class_2 = truth_image == 2;
    test_image_class_3 = test_image == 3;
    truth_image_class_3 = truth_image == 3;
    similarity_class_0(i) = dice(test_image_class_0, truth_image_class_0);
    similarity_class_1(i) = dice(test_image_class_1, truth_image_class_1);
    similarity_class_2(i) = dice(test_image_class_2, truth_image_class_2);
    similarity_class_3(i) = dice(test_image_class_3, truth_image_class_3);
end
% mean Dice score fore each class
mean(similarity_class_0)
mean(similarity_class_1)
mean(similarity_class_2)
mean(similarity_class_3)
% mean Dice score fore each class
std(similarity_class_0)
std(similarity_class_1)
std(similarity_class_2)
std(similarity_class_3)

%% how well did we perform overall?
output_directory = 'network_test';
imds = imageDatastore(fullfile(basedir, 'testing'));
pxdsTruth = pixelLabelDatastore(fullfile(basedir, 'testing_label'),classNames,pixelLabelIds);
pxdsResults = pixelLabelDatastore(output_directory,classNames,pixelLabelIds);
n_images = 65;
[cell_intensity_truth_total,cell_intensity_DL_total] =  evalTotalError(imds,pxdsTruth,pxdsResults,n_images,targetimsize);

save('error_estimation.mat','cell_intensity_truth_total','cell_intensity_DL_total');


%% fit result

x = cell_intensity_truth_total';
y = cell_intensity_DL_total';
[fitobj gof] = fit(x,y,'poly1');
plot(cell_intensity_truth_total,cell_intensity_DL_total,'kx')
hold all
xax = linspace(min(cell_intensity_truth_total),max(cell_intensity_truth_total),100);
plot(xax,fitobj.p1*xax +fitobj.p2 ,'r-','LineWidth',2)
xlabel('Ground truth fluorescence')
ylabel('Computed fluorescence')
text(60,120,['y = ' num2str(fitobj.p1) ' + ' num2str(fitobj.p2)])
hold off
%% boot strap slope

slope = [];
for i = 1:10000
    idx = randi([1 length(cell_intensity_truth_total)],1,100);
    x = cell_intensity_truth_total(idx);
    y = cell_intensity_DL_total(idx);
    fitobj = fit(x',y','poly1');
    slope(i) = fitobj.p1;
end

h = histogram(slope);
x = ((h.BinEdges(2:end) - h.BinEdges(1:end-1))/2) + h.BinEdges(1:end-1);
y = h.Values;
[curve fitobj gof] = fit(x',y','gauss1');
histogram(slope)
hold all
plot(curve)
text(0.2,750,['y = ' num2str(curve.a1) '*exp(-((x-' num2str(curve.b1) ')/' num2str(curve.c1) ')^2)'])