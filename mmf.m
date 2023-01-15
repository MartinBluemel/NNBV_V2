function [X,Y] = mmf(Input,r,M_T,modes_n)

original_image = imresize(Input,r/size(Input,1));

vec_input=decomposeModes(modes_n,original_image);
input_distribution=zeros(r,r);

for i=1:size(modes_n,1)
    
    input_distribution=input_distribution+vec_input(i)*squeeze(modes_n(i,:,:));
    
end
result_input_N = uint8(normalization(abs(original_image),0,255))*1;

%decompose
input_distribution_n=input_distribution./sum(sum(abs(input_distribution)));
mode_weights_input=decomposeModes(modes_n,input_distribution_n);

%  Propagate Through Fiber
mode_weights_output=M_T*vec_input;
Output_distribution=zeros(r,r);

for i_o=1:size(modes_n,1)
    Output_distribution=Output_distribution+mode_weights_output(i_o)*squeeze(modes_n(i_o,:,:));
end

Output_distribution_n=Output_distribution./sum(sum(abs(Output_distribution)));
result_output_N= uint8(normalization(abs(Output_distribution_n),0,255))*1;

X=result_output_N;
Y=result_input_N;
end

