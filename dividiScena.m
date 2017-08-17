
function [A B C D] = dividiScena(I)

    [M N] = size(I);
    
    A = I(1:N/2, 1:N/2);
    
    B = I(1:M/2, N/2+1:N);
    
    C = I(M/2+1:M, 1:N/2);
    
    D = I(M/2+1:M, N/2+1:N);
   
end