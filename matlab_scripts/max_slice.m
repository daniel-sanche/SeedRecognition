function count = max_slice(single_seed_shortname, num_seed)

root = '/Volumes/seagate_backup/raw_image_slice';

config = parse_slice_config('slice_config');

slice_folder = fullfile(root,config.(sprintf('%s',single_seed_shortname)){num_seed,1}{1});


temp = fullfile(slice_folder,'*.png');

files = dir([temp]);
count = 0;
for i = 1:size(files,1)
    filename = files(i).name;
    startIndex = regexp(filename,'[0-9]+.png');
    if ~isempty(startIndex)
        count = count+1;
    end
end


end
