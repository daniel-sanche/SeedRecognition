function config = parse_slice_config(config_file)
%find the correspoding slice number for the seed image
seed_shortname = {'bc'; 'bj'; 'bn'; 'sa'; 'bry';...
                  'brb'; 'cm'; 'cst'; 'cso'; 'sl';...
                  'cbp'; 'brc'; 'cd'; 'ds'; 'brp';...
                  'sf1'; 'sii1'; 'siv1'; 'sp1'; 'sv1';...
                  'cca'; 'cch'; 'cgr'; 'cme'; 'cpe';...
                  'ahy';'apacc';'apr';'apo';'are'};


fid = fopen(config_file);
out = textscan(fid, '%d%s', 'delimiter', ' ');
fclose(fid);

config = {};
for i = 1:size(out{1,1},1)
    config.(sprintf('%s', seed_shortname{ceil(i/10)})){mod(i-1,10)+1,1} = out{1,2}(i,1);
end

