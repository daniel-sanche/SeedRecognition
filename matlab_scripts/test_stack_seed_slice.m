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
    in_focus = slice_stack('nd1413', 1, 5, 5);

    filename = strcat('../mystack/','nd1413','.png');

    imwrite(in_focus,filename);
end
toc