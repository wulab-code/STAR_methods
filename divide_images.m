function divide_images(image_directory,ground_truth,training,validation,testing)

mkdir(fullfile(image_directory,'training'));
mkdir(fullfile(image_directory,'validation'));
mkdir(fullfile(image_directory,'testing'));
mkdir(fullfile(image_directory,'training_label'));
mkdir(fullfile(image_directory,'validation_label'));
mkdir(fullfile(image_directory,'testing_label'));

% assumes original images and ground truth images are named the same way. 

% find all png images
originals = dir(fullfile(image_directory,'*.png'));
disp('moving validation images...')
c = 1;
for i = 1:length(validation)
	copyfile(fullfile(image_directory,originals(validation(i)).name),...
        fullfile(image_directory,'validation',[zerostr(4,c) '.png']));
 	copyfile(fullfile(ground_truth,originals(validation(i)).name),...
        fullfile(image_directory,'validation_label',[zerostr(4,c) '.png']));
    c = c+1;
end
disp('moving testing images...')
c = 1;
for i = 1:length(testing)
    copyfile(fullfile(image_directory,originals(testing(i)).name),...
        fullfile(image_directory,'testing',[zerostr(4,c) '.png']));
	copyfile(fullfile(ground_truth,originals(testing(i)).name),...
        fullfile(image_directory,'testing_label',[zerostr(4,c) '.png']));
     c = c+1;
end
disp('moving training images...')
c = 1;
for i = 1:length(training)
	copyfile(fullfile(image_directory,originals(training(i)).name),...
        fullfile(image_directory,'training',[zerostr(4,c) '.png']));
    copyfile(fullfile(ground_truth,originals(training(i)).name),...
        fullfile(image_directory,'training_label',[zerostr(4,c) '.png']));
    c = c+1;
end
