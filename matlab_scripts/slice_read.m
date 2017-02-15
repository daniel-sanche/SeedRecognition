function out = slice_read(single_seed_shortname, num_seed, channel, num_slice)
% example usage: slice_read('bc', 1, 'rgb', 40);
% change the root path to the folder where raw image slices reside
root = '/Volumes/seagate_backup/raw_image_slice';

config = parse_slice_config('slice_config');

slice_RGB = imread(fullfile(root, config.(sprintf('%s', single_seed_shortname)){num_seed, 1}{1}, sprintf('%.3d.png', num_slice)));

if (size(slice_RGB,3) == 3)
    slice_HSV = rgb2hsv(slice_RGB);
    
    slice_GRAY = rgb2gray(slice_RGB);
elseif (size(seed_RGB,3) == 1)
    slice_GRAY = slice_RGB;
end



switch channel
    case 'r'
        out = im2double(slice_RGB(:,:,1));
        
    case 'g'
        out = im2double(slice_RGB(:,:,2));
        
    case 'b'
        out = im2double(slice_RGB(:,:,3));
        
    case 'h'
        out = im2double(slice_HSV(:,:,1));
        
    case 's'
        out = im2double(slice_HSV(:,:,2));
        
    case 'v'
        out = im2double(slice_HSV(:,:,3));
        
    case 'gray'
        out = im2double(slice_GRAY);
        
    case 'rgb'
        out = im2double(slice_RGB);
        
    case 'lab'
        %transoform to cielab space
        cform = makecform('srgb2lab');
        seed_LAB = applycform(slice_RGB, cform);
        out = im2double(seed_LAB);
    otherwise
        out = im2double(slice_GRAY);
end