% cellinfo.m
%
% load celldata structure with information about V1 single unit
% data recorded using natural vision movies.
%
% after running this script, celldata contains info for cells nn=1...12:
% celldata(nn).cellid - name of cell
% celldata(nn).rfdiameter - diameter of classical receptive field
%                           in pixels (in orginal stimulus files)
% celldata(nn).datafile - file with response data (binned at 14 ms)
%                         and downsampled (16 x 16 pix = 2 crf
%                         diameters) stimulus files
% celldata(nn).fullstimfile - full size timulus file with corresponding
%                             response in datafile
% celldata(nn).fullvalstimfile - full size validation stimulus file
%
%
% example:
% cellinfo;
% nn=1;
% fprintf('loading cell %s data from %s...\n',...
%         celldata(nn).cellid,celldata(nn).datafile);
% load(celldata(nn).datafile);
% fprintf('resp: %d x %d vector\n',size(resp));
% fprintf('stim: %d x %d x %d matrix\n',size(stim));
% fprintf('validation stim: %d x %d x %d matrix\n',size(vstim));
%
% last mod SVD 2005-12-03
%
cellids={'r0206B','r0208D','r0210A','r0211A','r0212B','r0217B',...
         'r0219B','r0220A','r0221A','r0222A','r0223A','r0225C'};
rfdiameter=[34 30 30 24 24 20 26 24 36 30 30 24];

for ii=1:12
   celldata(ii).cellid=cellids{ii};
   celldata(ii).rfdiameter=rfdiameter(ii);
   celldata(ii).datafile=sprintf('%s_data.mat',cellids{ii});
   celldata(ii).fullstimfile=sprintf('%s_stim.imsm',cellids{ii});
   celldata(ii).fullvalstimfile=sprintf('%s_valstim.imsm',cellids{ii});
end

