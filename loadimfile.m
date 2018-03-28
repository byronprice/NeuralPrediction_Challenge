% function mov = loadimfile(sImFile,startframe,endframe)
%
% load an imsm-format visual stimulus.
%
% parameters:
% sImFile - file name of imsm file
% startframe - first frame to load. default=1.
% endframe - last frame to load. default=end. (also end if endframe<startframe)
%
% returns:
% mov - X x X x T in size. X is the pixel diameter of each frame. T
%       is number of time bins
%
% created SVD 2005-12 (hacked from more complicated routines)
%
function mov = loadimfile(sImFile,startframe,endframe,scale_to_pix,mask_to_pix,crop_to_pix,smallfmt,dosmooth)

[fid, framecount, iconsize, iconside, filetype,altname] = openimfile(sImFile);
if filetype==-1,
   disp(sprintf('readimfile.m:  ERROR could not open %s',sImFile));
   mov=[];
   return;
end
if ~exist('startframe','var') | startframe <= 0,
   startframe=1;
end
if ~exist('endframe','var') | endframe <= 0,
   endframe=framecount;
end
if ~exist('scale_to_pix','var'),
   scale_to_pix=0;
end
if ~exist('mask_to_pix','var'),
   mask_to_pix=0;
end
if mask_to_pix>scale_to_pix,
   mask_to_pix=0;  % this is just to keep naming of pre-processed
                   % files from being redundant
end
if ~exist('crop_to_pix','var'),
   crop_to_pix=0;
end
if not(exist('smallfmt','var')),
   smallfmt=0;
end
if ~exist('dosmooth','var'),
   dosmooth=1;
end

if scale_to_pix==0,
   scale_to_pix=iconside(1);
end

%
% figure out if this imsm file has been loaded with these
% parameters before.
%
precompfile=sprintf('%s.%d.%d.%d.%d.%d',sImFile,scale_to_pix,...
                    mask_to_pix,crop_to_pix,smallfmt,dosmooth);
FASTLOAD=0;

if exist(precompfile,'file') & scale_to_pix<iconside,
   FASTLOAD=1;
   sImFile=precompfile;
   fclose(fid);
   [fid,framecount,iconsize,iconside,filetype,altname]=openimfile(sImFile);
   scale_to_pix=iconside(1);
elseif scale_to_pix<iconside,  % ie, hasn't been pre-shrunk but want to do it
   w=unix(['touch ',precompfile]);
   if ~w,
      w=unix(['\rm ',precompfile]); % delete to avoid collisions
      FASTLOAD=2;
      sf0=startframe;
      if framecount<endframe,
         ef0=framecount;
      else
         ef0=endframe;
      end
      startframe=1;
      endframe=framecount;
   end
end
if crop_to_pix==0,
   crop_to_pix=scale_to_pix;
end

fmtstr={'uint8','uint8','uint8','double','uint32'};
pixsize=[1 1 1 8 4];
%keyboard
if filetype==0,
   ps=1;
elseif filetype < 5,
   ps=pixsize(filetype);
else
   ps=1;
end
if startframe > 1 & startframe <= framecount,
   fseek(fid,iconsize*(startframe-1)*ps,0);
   if endframe >= startframe & endframe < framecount,
      framecountout = endframe-startframe+1;
   else
      framecountout = framecount-startframe+1;
   end
else
   if endframe > 0 & endframe < framecount,
      framecountout = endframe;
   else
      framecountout= framecount;
   end
end

fprintf('Reading movie file %s (frames %d-%d)...\n',...
        sImFile,startframe,endframe);
if crop_to_pix<iconside | scale_to_pix<iconside,
   [fmask,croprange]=movfmask(iconside(1),mask_to_pix/scale_to_pix, ...
			      crop_to_pix*iconside(1)./ ...
                              scale_to_pix);
   mov=readfromimfile(fid,framecountout,iconside,filetype,...
                      crop_to_pix,fmask,croprange,smallfmt,dosmooth);
else
   mov=readfromimfile(fid,framecountout,iconside,filetype,...
                      [],[],[],smallfmt,dosmooth);
end

fclose(fid);

if FASTLOAD==2,
   %fprintf('saving pre-comp stim: %s\n',precompfile);
   writeimfile(mov,precompfile,4);  % since its been resized, want
                                    % to preserve more than a byte
   if length(msize)==3,
      mov=mov(:,:,sf0:ef0);
   else
      mov=mov(:,sf0:ef0);
   end
end



% function[SmallMov] = movresize(BigMov,final_pix,fmask,croprange, 
%                                smallfmt,dosmooth)
%
%	Version 1: 1/99 BV
%	This version is intended as a tool for converting movies with a large
%	number of pixels to movies with a small number of pixels.
% 
%	This version relies on a user input value for the number of final
% 	pixels.
%
%	The down sampling assumes the input of a 3d array of shorts for the
%	BigMov. 
%	The output is a 3d array of doubles (SmallMov).
%
%       Modified 3/22/99 - SVD - add masking
%	Modified 4/1/99 - SVD - add crop around masked region
%	Modified 4/9/99 - SVD - add uint8 format return option
%       Modified 5/11/99 - SVD - add crop_to_pix option
%
%       Parameters:
%       BigMov - pix X pix X Tmax array of uint8
%       final_pix - final pix (after cropping)
%       fmask - pix X pix mask (can generate this with movfmask or
%               simply pass ones(pix,pix) to skip any masking
%       croprange - [xmin ymin xmax ymax] range to crop out of
%                   original pix X pix frames of BigMov
%       smallfmt - 0 (default), return doubles, scale to lie
%                     between -1 and 1
%                  1 return uint8
%       dosmooth   - (default 0) if 1, smooth to avoid aliasing--if necessary
%
function[SmallMov] = movresize(BigMov,final_pix,fmask,croprange,...
                               smallfmt,dosmooth)

if min(min(double(BigMov(:,:,1))))<0,
   BGPIX=20/128-1;  % grayscale color of bg, scaled to [-1,1] range
else
   BGPIX=20;
end

% Step 1: Enter the output number of pixels

if not(exist('smallfmt','var')),
   smallfmt=0;
end
if ~exist('dosmooth','var'),
   dosmooth=0;
end

original_pix_per_side = size(BigMov, 1);

% croprange should define a square region!
mask_pix=croprange(3)-croprange(1)+1;
resize_pix=final_pix;
%resize_pix=round(mask_pix*scaleby);


% Step 2: Create the mesh of grid points to feed to the interp command

if croprange(1)<=0,
   cropovershoot=1-croprange(1);
   sm_resize_pix=round((croprange(4)-croprange(2)+1-2*cropovershoot) / ...
       (croprange(4)-croprange(2)+1) * resize_pix /2) *2;
   zeropad=(resize_pix-sm_resize_pix)/2;
   x = linspace(croprange(2)+cropovershoot, ...
                croprange(4)-cropovershoot, sm_resize_pix);
   y = linspace(croprange(1)+cropovershoot, ...
                croprange(3)-cropovershoot, sm_resize_pix);
else
   zeropad=0;
   x = linspace(croprange(2), croprange(4), resize_pix);
   y = linspace(croprange(1), croprange(3), resize_pix);
end
[X,Y] = meshgrid(x,y);

if dosmooth & resize_pix/mask_pix < 1,  % need to smooth
   nn=min([round(mask_pix/resize_pix)*2+1 11]);
   h1 = DesignFilter(nn-1,resize_pix/mask_pix);
   
   %a = filter2(h1',filter2(h1,A)); 
else
end

% Step 3: Loop through BigMov and convert one frame at a time

temp_count = 0;
hundred_count = 0;

num_frames = size(BigMov,3);

SmallMov = zeros(resize_pix, resize_pix, num_frames);

% SVD - add 4/9/99
% if specified, convert back to uint8
if smallfmt==1,
   SmallMov=uint8(SmallMov);
end

for i = 1:num_frames,

   % Get frame
   framesmall = BigMov(:,:,i);
   
   % Convert to doubles and add padding if necessary to match
   % crop_to_pix
   frame=double(framesmall);
   %keyboard
   
   % 4/3/01 - temporarily turn off anti-aliasing smoothing --makes
   % natural stuff worse????!?!??!
   if dosmooth & resize_pix/mask_pix < 1,  % need to smooth
      frame = rconv2(frame,h1'*h1); 
   end
   
   % mask out unwanted regions
   % SVD - add 3/22/99
   frame = frame .* fmask + (1-fmask) .* BGPIX;
   
   % somehow deal with the fact that crop range might be
   % outside of the boundaries of the original movie!
   
   % Run interp
   if zeropad,
      small_frame = ones(resize_pix,resize_pix)*...
	   mean([frame(1,[1 end]) frame(end,[1 end])]);
      small_frame(ceil(zeropad)+1:resize_pix-floor(zeropad),...
                  ceil(zeropad)+1:resize_pix-floor(zeropad))= ...
         interp2(frame, X, Y);
   else
      %frame = filter2(h1',filter2(h1,frame)); 
      small_frame = interp2(frame, X, Y);
   end
   
   % SVD - add 4/9/99
   % if specified, convert back to uint8
   if smallfmt==1,
      small_frame=uint8(small_frame);  
   end
   
   % Pack output
   % modified SVD 4/1/99 to add cropping feature
   SmallMov(:,:,i) = small_frame;
   
   temp_count = temp_count + 1;
end




%------------------------------------------------------------------------------
% This function converts from shorts to doubles
% Assumes a 0 to 255 range and converts to a -1/+1 range.
%------------------------------------------------------------------------------
function[out] = Convert2Double(In)

temp = double(In);

out = (temp/128) -1 ;

% Note this function is much easier since the converstion from
% shorts to doubles doesn't have to worry about 'wrap-around' of
% values


%------------------------------------------------------------------------------
% This function converts from doubles to shorts
% Assumes a -1/+1 range for In and converts to a 0 to 255 range.
% inserted SVD 4/9/99
%------------------------------------------------------------------------------
function[out] = Convert2Uint8(In)

out = uint8((In+1)*128);



% code for DesignFilter stolen from Matlab's imresize.m

function b = DesignFilter(N,Wn)
% Code from SPT v3 fir1.m and hanning.m

N = N + 1;
odd = rem(N,2);
wind = .54 - .46*cos(2*pi*(0:N-1)'/(N-1));
fl = Wn(1)/2;
c1 = fl;
if (fl >= .5 | fl <= 0)
    error('Frequency must lie between 0 and 1')
end 
nhlf = fix((N + 1)/2);
i1=1 + odd;

if odd
   b(1) = 2*c1;
end
xn=(odd:nhlf-1) + .5*(1-odd);
c=pi*xn;
c3=2*c1*c;
b(i1:nhlf)=(sin(c3)./c);
b = real([b(nhlf:-1:i1) b(1:nhlf)].*wind(:)');
gain = abs(polyval(b,1));
b = b/gain;




% function mov = readfromimfile(fid,framestoread,iconside,filetype,
%                               final_pix,fmask,croprange,smallfmt,dosmooth)
%
% Last update: SVD 11/30/99
% Last update: SVD 3/19/02 - added support for arbitrary number of
%                            dimensions and uint8/double/uint32 resolution
%
% Required parameters:
%
% fid          - matlab binary movie file id (use openimfile to generate it)
% framestoread - number of frames to read
% iconside     - vector of the size of each spatial dimension
% filetype     - code to identify type of file:
%                  0. old (SGI) .im file ("big-endian" integer format)
%                  1 or 2. 1-byte-per-pixel grayscale (.imsm)
%                  (1="big-endian", 2="little-endian" integer format)
%                  3. double
%                  4. uint32
%
% Optional parameters:
%
% final_pix    - scale cropped region of movie to final_pix X final_pix
%                (default=iconside)
% fmask        - iconside X iconside 1/0 matrix to mask movie
%                before scaling (default=ONES(iconside,iconside))
% croprange    - coordinates in movie frame to crop out before scaling
%                of the format [x0 y0 x1 y1].  
%                (default=[1 1 iconside iconside])
% smallfmt     - flag.  if 1, return mov as uint8.  if 0, return mov
%                as double (default=0)
% 
% Returns:
%
% mov          - size X size X ... X time matrix containing frames from
%                the movie
%
function mov = readfromimfile(fid,framestoread,iconside,filetype,final_pix,fmask,croprange,smallfmt,dosmooth)

if nargin < 4,
   disp ('readfromimfile.m:  ERROR: must supply first 4 arguments.');
   return
end
if ~exist('dosmooth','var'),
   dosmooth=0;
end

% set parameters to default values if they haven't been specified
if not(exist('croprange','var')),
    [fmask,croprange]=movfmask(iconside(1),1.0,iconside(1));
    fmask=ones(iconside(1));
end
if not(exist('final_pix','var')) | isempty(final_pix),
   final_pix=iconside(1);
   redosize=0;
elseif final_pix==iconside & isempty(find(fmask<1)),
   redosize=0;
else
   redosize=1;
end
if not(exist('smallfmt','var')),
    smallfmt=0;
end

% set scaling ratio
%ratio=scale_to_pix./iconside(1);
iconsize=iconside(1).^2;  % total pixels per frame

% determine final dimensions, ie, scaled cropped region for setting
% size of mov
%crop_to_pix=round((croprange(3)-croprange(1)+1).*ratio);
crop_to_pix=final_pix;
if crop_to_pix < iconside(1),
   mov=zeros(crop_to_pix,crop_to_pix,framestoread);
elseif ismember(filetype,[3 4 5 105]),
   mov=zeros([iconside(:); framestoread]');
else
   mov=zeros(crop_to_pix,crop_to_pix,framestoread);
end
if smallfmt==1,
   mov=uint8(mov);
end

if ismember(filetype,[1 3 4]),
   fmtstr={'uint8','uint8','uint8','double','uint32'};
   
   % load in chunks to save memory
   CHUNKSIZE=500;
   pixsofar=0;
   for ii=1:ceil(framestoread/CHUNKSIZE),
      if ii==ceil(framestoread/CHUNKSIZE),
         f2r=mod(framestoread-1,CHUNKSIZE)+1;
      else
         f2r=CHUNKSIZE;
      end
      [tmov,n]=fread(fid,[prod(iconside) f2r],fmtstr{filetype});
      iirange=(ii-1)*CHUNKSIZE+(1:(n./prod(iconside)));
      if redosize,
         tmov=reshape(tmov,[iconside(:)' f2r]);
         mov(:,:,iirange)=movresize(tmov,final_pix,fmask,croprange,...
                               smallfmt,dosmooth);
      elseif smallfmt,
         mov(:,:,iirange)=uint8(tmov);  
      %elseif filetype==1,
      %   mov(:,:,iirange)=(double(tmov)/128)-1;
      else
         mov((pixsofar+(1:n))')=tmov(:)';
      end
      pixsofar=pixsofar+n;
   end
elseif ismember(filetype,[2]), % imsm file format
   
   % load in chunks to save memory
   % adapted from above, SVD 5/9/03
   
   CHUNKSIZE=500;
   pixsofar=0;
   iconside=[iconside iconside];
   for ii=1:ceil(framestoread/CHUNKSIZE),
      if ii==ceil(framestoread/CHUNKSIZE),
         f2r=mod(framestoread-1,CHUNKSIZE)+1;
      else
         f2r=CHUNKSIZE;
      end
      
      % load the next CHUNKSIZE (or whatever's left) from disk
      [tmov,n]=fread(fid, [prod(iconside) f2r],'uint8');
      
      % transpose each frame
      tmov=reshape(tmov,[iconside(:)' f2r]);
      tmov=permute(tmov,[2 1 3]);
      iirange=(ii-1)*CHUNKSIZE+(1:(n./prod(iconside)));
      
      if redosize,
         
         % mask and scale the current chunk, then save to mov matrix
         tmov=double(tmov);
         mov(:,:,iirange)=movresize(tmov,final_pix,fmask,croprange,...
                               smallfmt,dosmooth);
      elseif smallfmt,
         
         % just save the byte-sized data
         mov(:,:,iirange)=tmov;  
      else
         
         % output is double; file was uint8. so we need to convert
         % for output to mov
         mov((pixsofar+(1:n))')=double(tmov(:)');
      end
      pixsofar=pixsofar+n;
   end
   
else
   %
   % old imsm/im formats.
   % load and process each frame from the .im file, one at a time
   for ii=1:framestoread,
      if filetype == 0,  % im file format
         [nextframe, fcount] = fread(fid, [iconside,iconside], 'uint32');
         if fcount~=iconsize,
            disp('full frame not read!');
         end
         
         r = bitand(nextframe, 255);
         g = bitand(bitshift(nextframe,-8), 255);
         b = bitand(bitshift(nextframe,-16), 255);
         
         % deal with border regions?  check out DeFlag() in LoadinMov.m    
         %    v = ((r * 0.3086) + (g * 0.6094) + (b * 0.082)) .* (fmask==1) + ...
         %        (-999.99) .* (fmask==0);
         
         % convert from RGB to grayscale
         v = round((r .* 0.3086) + (g .* 0.6094) + (b .* 0.082));
         
         % clean up:    any pixels > 255 set to 255, < 0 set to 0
         good_mask = (v > 0) & (v <= 255);
         good_temp = v .* good_mask;
         
         high_mask = v > 255;
         high_temp = high_mask * 255;
         
         % note use of transpose!  this is to match bill's LoadInMov.m routine!
         vout = uint8(good_temp + high_temp)';
      
      elseif filetype>=101 & filetype<=104, % complex phase format
         [nextframe, fcount] = fread(fid, [iconside,iconside], 'double');
         if fcount~=iconsize,
            disp('full frame not read!');
            keyboard
         end
         %	nextframe=double(nextframe);
         %        nextframe
         %	pause
         
         % no transpose here! it's already been done!
         vout = nextframe;
         if filetype==102 | filetype==104,
            vout=vout*i;
         end
         if filetype==103 | filetype==104,
            vout=-vout;
         end
         
      elseif filetype==105,
         %keyboard
         [nextframe,fcount]=fread(fid,prod(iconside),'double');
         %if fcount<prod(iconside),
         %   keyboard
         %end
         mov(:,:,:,ii)=reshape(nextframe,iconside');
         
      end
      
      if filetype~=105,
         if redosize,
            mov(:,:,ii)=movresize(vout,final_pix,fmask,croprange,...
                                  smallfmt,dosmooth);
         elseif smallfmt,
            mov(:,:,ii)=uint8(vout);  
         else
            mov(:,:,ii)=double(vout);
            %mov(:,:,ii)=(double(vout)/128)-1;
         end
      end
   end
end


% function [fid,framecount,iconsize,iconside,filetype,altname]=openimfile(sImFile)
%
% Last update: SVD 11/30/99
%
% Required parameters:
%
% sImFile      - file name of movie for input (.im or .imsm)
%
% Returns:
%
% fid          - matlab binary movie file id (use openimfile to generate it)
% iconside     - pixels per side of each frame in the movie file
% iconsize     - pixels per frame (iconside^2... duh)
% filetype     - code to identify type of file:
%                  0. standard .im file
%                  1 or 2. 1-byte-per-pixel grayscale (.imsm)
%                  format
% filetype:    - which type of known movie file you've just opened
%                 -1 - error opening file
%                  0 - im file format (raw RGB - uint32 per pixel)
%                  1/2/3 - imsm file format (grayscale - uint8 per pixel)
%                  4 - imsm file format double
%                  5 - imsm file format uint32
% altname:     - if imfile loaded from a remote network location 
%                (ie, checkbic>0) name of temp file
%
function [fid,framecount,iconsize,iconside,filetype,altname]=openimfile(sImFile)

if not(exist('sImFile','var')),
   disp('syntax: openimfile(sImFile)');
   framecount=-1;
   iconsize=-1;
   iconside=-1;
   filetype = -1;
   return
end

altname=sImFile;

% open file using "little-endian" format.
[fid,sError]=fopen(sImFile,'r','l');

if fid >= 0,
   framecount=fread(fid,1,'uint32');
   iconsize=fread(fid,1,'uint32');
   if framecount==0 & iconsize==0,
      filetype=2;
      framecount=0;  % must read big-endian uint32s
      for ii=1:4,
         framecount=bitshift(framecount,8)+fread(fid,1,'uint8');
      end
      iconsize=0;
      for ii=1:4,
         iconsize=bitshift(iconsize,8)+fread(fid,1,'uint8');
      end
      iconside=sqrt(iconsize);
      
   elseif framecount==0 & iconsize==1,
      filetype=2;
      framecount=fread(fid,1,'uint32');  %little-endian uint32
      iconsize=fread(fid,1,'uint32');
      iconside=sqrt(iconsize);
   elseif framecount==0 & ismember(iconsize,[3 4 5]),
      filetype=iconsize;
      %arms=[0 imfilefmt framecount spacedimcount iconsizeout imfilefmt];
      framecount=fread(fid,1,'uint32');
      spacedimcount=fread(fid,1,'uint32');
      iconside=fread(fid,spacedimcount,'uint32');
      iconsize=prod(iconside);
   elseif framecount==0 & (iconsize>=101 & iconsize<=104),
      filetype=iconsize;
      framecount=fread(fid,1,'uint32');  %little-endian uint32
      iconsize=fread(fid,1,'uint32');
      iconside=sqrt(iconsize);
      
   elseif framecount==0 & iconsize==105,  % log sf format
      filetype=105;
      disp('Opening lim file...');
      framecount=fread(fid,1,'uint32');  %little-endian uint32
      iconside=fread(fid,3,'uint32');
      iconsize=prod(iconside);
      
   else
      fclose(fid);
      [fid,sError]=fopen(sImFile,'r','b');
      framecount=fread(fid,1,'uint32');
      iconsize=fread(fid,1,'uint32');
      iconside=sqrt(iconsize);
      
      filetype=0;
   end

else
   framecount=-1;
   iconsize=-1;
   iconside=-1;
   filetype = -1;
end

% function [fmask,crop]=movfmask(stimpix,ratio,croppix,edgestyle);
%
% ratio=fraction of circle of diameter stimpix to mask in.  returns 1's inside
% circle and 0's outside
%
% edgestyle (default 0) 0 - gaussian mask with sigma = ratio
%                       1 - linear ramp to bg from ratio to 1.2*ratio
%
function [fmask,crop]=movfmask(stimpix,ratio,croppix,edgestyle);

if not(exist('croppix','var')),
   croppix=stimpix;
end
if not(exist('edgestyle','var')),
   edgestyle=0;
end

%SURROUND=1.2;
SURROUND=0.9;

if ratio <= 1 & ratio>0,
   [mx,my]=meshgrid(1:stimpix,1:stimpix);
   c=round((stimpix+1)/2);
   d=sqrt((c-mx).^2+(c-my).^2)./(c-2);
   
   if edgestyle==0,
      % try a gaussian mask with 1 sigma at ratio radius.
      sigma=ratio;
      fmask=exp(-(d./(2*sigma)).^2);
      fmask=fmask./max(fmask(:));
   
   else   
      % alternatives that don't seem to work very well
      %fmask=(d <= ratio) + (SURROUND-d./ratio)./(SURROUND-1) .* ...
            (d > ratio & d<=ratio*SURROUND);
      fmask=(d <= ratio*SURROUND) + (1-d./ratio)./(1-SURROUND) .* (d > ratio*SURROUND & d<=ratio);
      %fmask=(d <= ratio);
   end
else
   fmask=ones(stimpix,stimpix);
end
      
%figure(2);
%imagesc(fmask);
%pause

crop_start=floor((stimpix-croppix)/2)+1;
crop_stop=floor((stimpix+croppix)/2);
if crop_start < 1,
   crop_start=1-(floor((stimpix+croppix)/2) - stimpix);
end

crop=[crop_start,crop_start,crop_stop,crop_stop];
