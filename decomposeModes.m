function [a_indices]=decomposeModes(modes,Input)

for i=1:size(modes,1)
    A_k=sum(sum(squeeze(conj(modes(i,:,:))).*Input));
    K=sum(sum(squeeze(modes(i,:,:)).*conj(squeeze(modes(i,:,:)))));
    a_indices(i,1)=A_k/K;
end