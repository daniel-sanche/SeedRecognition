function in_focus = slice_stack(folder_name, range_start_dist, range_end_dist, interval)

root = '/home/sanche/Datasets/Seed_Images';

config = parse_slice_config('/home/sanche/Datasets/Seed_Images/slice_config');

slice_folder = [root, '/', folder_name];

temp = fullfile(slice_folder, '*.png');

files = dir([temp]);

count = 1;

for i = 1:size(files,1)
    filename = files(i).name;
    startIndex = regexp(filename,'[0-9]+.png');
    if ~isempty(startIndex)
        slice_files{count} = filename;
        count = count +1;
    end
    
end

num_images = length(slice_files);
num_end = num_images - range_end_dist;
num_start = 1 + range_start_dist;

count = 1;

for i = num_start:interval:num_end
    images{count} = fullfile(slice_folder,slice_files{i});
    count = count +1;
end

in_focus = fstack_lbp_incre_c(images);

% imwrite(in_focus, fullfile(slice_folder,'stack.png'));





