clc;
clear;
seed_shortname = {'bc'; 'bj'; 'bn'; 'sa'; 'bry'; 'brb';  'cm'; 'cst'; 'cso'; 'sl'; 'cbp'; 'brc'; 'cd'; 'ds'; 'brp'...
    ; 'sf1'; 'sii1'; 'siv1'; 'sp1'; 'sv1'...
    ; 'cca'; 'cch'; 'cgr'; 'cme'; 'cpe'...
    ;'ahy';'apacc';'apr';'apo';'are'};

% stack the first sample in each species, from the top image slice to the
% bottom slice with an interval of 1.
tic
for i = 1:30
    in_focus = slice_stack(seed_shortname{i}, 1, max_slice(seed_shortname{i}, 1), 1, -1);

    filename = strcat('./mystack/',seed_shortname{i},'.jpg');

    imwrite(in_focus,filename);
end
toc