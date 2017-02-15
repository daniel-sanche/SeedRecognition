clc
clear

root = '/Volumes/seagate_backup/Raw image files from microscope system/acquired sept 2013/';

files = dir(strcat(root,'*.nd2'));

for i = 1:size(files)
    file = strcat(root, files(i).name);
    
    [pathstr, filename, ext] = fileparts(file);
    
    data = bfopen(file);
    
    %-------------------------- save image slice ------------------------------
    series1 = data{1, 1};
    
    num_slices = size(series1,1)/3;
    
    directory = strcat(root, filename);
    
    if (exist(directory, 'dir') == 0), mkdir(directory); end
    
    
    
    for i = 1:num_slices
        %bgr I guess
        %determine whether it's uint16 or uint8. photographer seems to mix
        %it
        if isa(series1{1,1}, 'uint16')
            temp(:,:,3) = uint8(series1{(i-1)*3+1,1}/256);
            temp(:,:,2) = uint8(series1{(i-1)*3+2,1}/256);
            temp(:,:,1) = uint8(series1{(i-1)*3+3,1}/256);
        else
            temp(:,:,3) = uint8(series1{(i-1)*3+1,1});
            temp(:,:,2) = uint8(series1{(i-1)*3+2,1});
            temp(:,:,1) = uint8(series1{(i-1)*3+3,1});
        end
        image_num = sprintf('%.3d', i);
        file_name = strcat(directory,'/', image_num, '.png');
        
        temp_resize = imresize(temp, [1280 1600]);
        % the technician move the whole image to the left 36px.
        temp_resize = temp_resize(1:1280,37:1600,:);
        temp_resize = temp_resize(1:1280,1:1400,:);
        slice = imresize(temp_resize, 0.5);
        
        imwrite(slice, file_name, 'png');
        
        
        
    end
    
end

