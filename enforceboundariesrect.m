function newbound = enforceboundariesrect(oldbound,imsize)
if oldbound(1) < 1
    newbound(1) = 1;
else
    newbound(1) = oldbound(1);
end

if oldbound(2) < 1
    newbound(2) = 1;
else
    newbound(2) = oldbound(2);
end

if oldbound(1) + oldbound(3) > imsize(2)
%     newbound(3) = imsize(2)-newbound(1);
    newbound(3) = oldbound(3)-1;
else 
    newbound(3) = oldbound(3);
end

if oldbound(2) + oldbound(4) > imsize(1)
%     newbound(4) = imsize(1)-newbound(3);
    newbound(4) = oldbound(4) - 1;
else 
    newbound(4) = oldbound(4);
end