function [ accuracy ] = get10Accuracy( dist )
dist=mat2gray(dist);

for i=1:size(dist, 1)
    r=dist(i, :);
    [sr, idx]=sortrows(r(:));
    sortDist(i, :)=sr';
    sortDistIdx(i, :)=idx';
end

% from top-ranking to 10-th ranking
for i=1:10
    retrived=sortDistIdx(:, 1:i);
    rowSize=size(retrived, 1);
    hitCount=0;
    for j=1:rowSize
        row=retrived(j, :);
        hitCount=hitCount+sum(row==j);
    end
    accuracy(i)=hitCount/rowSize;
end
accuracy;

end

