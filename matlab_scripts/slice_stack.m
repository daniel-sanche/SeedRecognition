function in_focus = slice_stack(folder_name, start_img, num_images, interval)

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

total_images = length(slice_files);
count = 1;

for i = start_img:(start_img+num_images-1)
    idx = start_img+((i-1)*interval);
    if idx < total_images
        images{count} = fullfile(slice_folder,slice_files{idx});
        count = count +1;
    end
end

in_focus = fstack_lbp_incre_c(images);

% imwrite(in_focus, fullfile(slice_folder,'stack.png'));





