function out = fstack_lbp_incre_c(Scene, varargin)
%Parse inputs:
Options = ParseImdata(Scene);
Options = ParseInputs(Options, varargin{:});
M = Options.Size(1);
N = Options.Size(2);
P = Options.Size(3);

%********* Read images and compute fmeasure **********
%Initialize:
FM = zeros(M,N,2);

ImagesR = zeros(M,N,2);
ImagesG = zeros(M,N,2);
ImagesB = zeros(M,N,2);


%Read:
senstivity = 0.008;
scale = 11;
thresh_update = 0.00;


for p = 1:P
    %flag for update or not
    update = 0; 
    Im = imread(Scene{p});
    if size(Im,3) == 1
        Im = repmat(Im,[1,1,3]);
    end
    
    if p == 1
        ImagesR(:,:,1) = Im(:,:,1);
        ImagesG(:,:,1) = Im(:,:,2);
        ImagesB(:,:,1) = Im(:,:,3);
        ImagesR(:,:,2) = zeros(M,N,1);
        ImagesG(:,:,2) = zeros(M,N,1);
        ImagesB(:,:,2) = zeros(M,N,1);
        
%         using cv.lbpSharpness(Im, scale, senstivity) if you have the mex implementation
        bm = lbpSharpness(Im, scale, senstivity);
        temp = transform(im2double(bm));
        
        FM(:,:,1) = temp;
        FM(:,:,2) = zeros(M,N,1);
        update = 1;
    elseif p == 2
        ImagesR(:,:,1) = out(:,:,1);
        ImagesG(:,:,1) = out(:,:,2);
        ImagesB(:,:,1) = out(:,:,3);
        ImagesR(:,:,2) = Im(:,:,1);
        ImagesG(:,:,2) = Im(:,:,2);
        ImagesB(:,:,2) =Im(:,:,3);
              
%         using cv.lbpSharpness(Im, scale, senstivity) if you have the mex implementation
        bm = lbpSharpness(Im, scale, senstivity);
        temp = transform(im2double(bm));
        
        
        FM1 = FM(:,:,1);
        FM2 = temp;
        dif = abs(FM1 - FM2)/(M*N);
%        fprintf('%5.5f\n',sum(dif(:)));
        if sum(dif(:)) > thresh_update
            update = 1;
            FM(:,:,2) = temp;
        end
    else
        ImagesR(:,:,1) = out(:,:,1);
        ImagesG(:,:,1) = out(:,:,2);
        ImagesB(:,:,1) = out(:,:,3);
        ImagesR(:,:,2) = Im(:,:,1);
        ImagesG(:,:,2) = Im(:,:,2);
        ImagesB(:,:,2) = Im(:,:,3);
        
%         using cv.lbpSharpness(Im, scale, senstivity) if you have the mex implementation
        bm = lbpSharpness(Im, scale, senstivity);
        temp = transform(im2double(bm));
        if p == P
            temp = temp +1e-3;
        end
        

       

        FM1 = FMn;
        FM2 = temp;
                
        dif = abs(FM1 - FM2)/(M*N);
%         fprintf('%5.5f\n',sum(dif(:)));
        if sum(dif(:)) > thresh_update
            FM(:,:,1) = FMn;
            FM(:,:,2) = temp;
            update = 1;
            
        end
        
    end
    
    
    if (update == 1)
        FMn = sum(FM,3);
        if p == 1
            out(:,:,1) = uint8(sum((ImagesR), 3));
            out(:,:,2) = uint8(sum((ImagesG), 3));
            out(:,:,3) = uint8(sum((ImagesB), 3));

        else
            out(:,:,1) = uint8(sum((ImagesR.*FM), 3)./FMn);
            out(:,:,2) = uint8(sum((ImagesG.*FM), 3)./FMn);
            out(:,:,3) = uint8(sum((ImagesB.*FM), 3)./FMn);
           

        end
%         imshow(FM(:,:,2),[]);
        imshow(out);
    end
    
    %     fprintf('\b\b\b\b\b[%2.0i%%]',round(100*p/P))
    
    %     pause(0.1);
end

% fprintf('[100%%]\n')

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Options = ParseInputs(Opts,varargin)
Options = Opts;
N = length(varargin);
Options.Alpha = 0.1;
Options.Sth = 13;
Options.Focus = 1:Opts.Size(3);
Options.WSize = 9;
Params = {'alpha','sth','focus',...
    'fsize'};
if (N==1)
    eid = 'DepthMap:InputCheck';
    warning(eid, 'missing parameter, setting to deffault')
    return
end
for n = 1:2:N
    if ~ischar(varargin{n})
        error('Please check input arguments. Probably one value is missing')
    elseif isempty(strmatch(lower(varargin{n}), Params))
        error('Unknown option %s.', upper(varargin{n}))
    elseif (N < n + 1)
        error('Value for option %s is missing', upper(varargin{n}))
    else
        Select = strmatch(lower(varargin{n}), Params, 'exact');
        str = Params{Select};
        Value = varargin{n+1};
    end
    if strcmpi(str,'fsize')
        if ~isnumeric(Value)
            error('Value for parameter FOCUSSIZE must be numeric');
        elseif numel(Value)~=1
            error('Value for parameter FOCUSSIZE must be a scalar');
        else
            Options.WSize = Value;
        end
    elseif strcmp(str,'focus')
        if ~isnumeric(Value)
            error('FOCUS vector must be numeric')
        elseif numel(Value)~=Opts.Size(3)
            error('FOCUS vector must have %s elements',Opts.Size(3))
        else
            Options.Focus = Value;
        end
    elseif strcmp(str,'alpha')
        if ~isnumeric(Value)
            error('ALPHA must be numeric')
        elseif (Value>1)||(Value<=0)
            error('ALPHA must be in (0,1]')
        else
            Options.Alpha = Value;
        end
    elseif strcmp(str, 'sth')
        if ~isnumeric(Value)
            error('STH must be numeric')
        else
            Options.Sth = Value;
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Options = ParseImdata(Imdata)
P = length(Imdata);
if ~iscell(Imdata)
    error('First input must be a cell array')
end

Options.RGB = false;
Options.STR = false;
if ischar(Imdata{1})
    Options.STR = true;
    if ~exist(Imdata{1},'file')
        error('File %s doesnt exist!',Imdata{1})
    else
        %Im = imread(Imdata{1});
        Im = imread(Imdata{1});
        Options.Size(1) = size(Im, 1);
        Options.Size(2) = size(Im, 2);
        Options.Size(3) = P;
    end
else
    Im = Imdata{1};
end
if (ndims(Im)==3)
    Options.RGB = true;
elseif (ndims(Im)~=2)
    error('Images must be RGB or grayscale!')
end
end


function output = transform(input)
    output = 1*input.^3;

end