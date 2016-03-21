-- protocol: batchsize, inputPlane, outputPlane, kernelWidth, kernelHeight, inputWidth, inputHeight, strideWidth, strideHeight, Output, Input, Gradacc. 
metaTest = {}

require 'nn'
require 'sys'
require 'cunn'
require 'ccn2'
require 'cudnn'
cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
cudnn.verbose = false
require 'fbcunn'
require ("metahard")
-- require 'nnbhwd'

local SpatialConvolutionMetaHard, parent = torch.class('nn.SpatialConvolutionMetaHard', 'nn.Module')

transpose1 = nn.Transpose({1,2}, {2,3}, {3,4}):cuda()
transpose2 = nn.Transpose({4,1},{4,2},{4,3}):cuda()

function SpatialConvolutionMetaHard:__init(nInputPlane, nOutputPlane,
                                        kW, kH, dW, dH,iW,iH,bS)

   parent.__init(self) 
   ni = nInputPlane
   no = nOutputPlane
   kw = kW
   kh = kH
   dw = dW
   dh = dH
   ih = iH
   iw = iW
   bs = bS
   ccn =  ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
   self.playOutput = ccn
   self.playGradInput = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()
   self.playGradPara = ccn

   print(self.playOutput)
   print(self.playGradInput)
   print(self.playGradPara)
end

function SpatialConvolutionMetaHard:updateOutput(input)
   if torch.typename(self.playOutput) == 'ccn2.SpatialConvolution' then
      input2 = transpose1:updateOutput(input)
      out = self.playOutput:updateOutput(input2)
      print(input2:size())
      return transpose2:updateOutput(out)
   else
      return self.playOutput:updateOutput(input)
   end
end

function SpatialConvolutionMetaHard:updateGradInput(input, gradOutput)
   if torch.typename(self.playGradInput) == 'ccn2.SpatialConvolution' then
      input2 = transpose1:updateOutput(input)
      gradOutput2 = transpose2:updateGradInput(input2, gradOutput)
      gradOutput3 = self.playGradInput:updateGradInput(input2, gradOutput2)
      gradOutput4 = transpose1:updateGradInput(input2, gradOutput3)
      return gradOutput4
   else
      return self.playGradInput:updateGradInput(input, gradOutput)
   end
end

function SpatialConvolutionMetaHard:accGradParameters(input, gradOutput)
   if torch.typename(self.playGradPara) == 'ccn2.SpatialConvolution' then
      input2 = transpose1:updateOutput(input)
      gradOutput2 = transpose2:updateGradInput(input2, gradOutput)
      self.playGradInput:accGradParameters(input2, gradOutput2)
   else
      self.playGradPara:accGradParameters(input, gradOutput)
   end
end

function SpatialConvolutionMetaHard:reset()
   self.play:reset()
end

return metaTest
