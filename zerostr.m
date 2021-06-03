function out = zerostr(len,j)
% out = zerostr(len,j)
% len = length of string
% j is number

strlen = strlength(num2str(j));

out = num2str(j);

numtimes = len-strlen;

for i = 1:numtimes
%     zeronum = '0';
%     zeronum = strcat('0',zeronum);
    out = strcat('0',out);
end

% out = strcat(zeronum,num2str(j));
