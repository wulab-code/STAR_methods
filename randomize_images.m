function [training, validation, testing] = randomize_images(num_images,frac_training)

idx = num_images; % number of ground truth images
vec = 1:idx; % this variable will contain indices for the validation and testing images
for i = 1:round(frac_training*idx) 
    r = round(length(vec)*(rand));
    if r > length(vec)
        r = length(vec);
    elseif r < 1
        r = 1;
    end
    training(i) = vec(r); % this variable will contain indices for the training images
    vec(r) = [];
end

for i = 1:round(length(vec)/2) 
    r = round(length(vec)*(rand));
    if r > length(vec)
        r = length(vec);
    elseif r < 1
        r = 1;
    end
    testing(i) = vec(r); % this variable will contain indices for the training images
    vec(r) = [];
end

validation = vec; % contains indices for images that will be used in validation
