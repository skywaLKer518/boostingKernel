function [w] = updateWeigts(w,y,pred,alpha)


% keyboard
w = w .* exp(alpha* (pred~=y'));
w = w / sum(w);
% fprintf('\tweights updated! alpha = %f\n',alpha)