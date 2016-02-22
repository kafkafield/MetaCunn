—- protocol: batchsize, inputPlane, outputPlane, kernelWidth, kernelHeight, inputWidth, inputHeight, strideWidth, strideHeight, Output, Input, Gradacc. 
metaHard = {}

require 'nn'
require 'sys'
require 'cunn'
require 'ccn2'
require 'cudnn'
cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
cudnn.verbose = false
require 'fbcunn'
-- require 'nnbhwd'

local SpatialConvolutionMetaFFT, parent = torch.class('nn.SpatialConvolutionMetaFFT', 'nn.Module')


function SpatialConvolutionMetaFFT:writefile(filename, info)
	local wfile=io.open(filename, "w+”)
	assert(wfile)
	wfile:write(info)
	wfile:close()
end

function SpatialConvolutionMetaFFT:__init(nInputPlane, nOutputPlane,
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
   mods = {}
   mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh):cuda()
   mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
   mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
   mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()
   timeOut = {}
   timeGradInput = {}
   timeGradPara = {}
   local steps = 3
   for j=1,#mods do

      collectgarbage()
      if torch.typename(mods[j]) == 'ccn2.SpatialConvolution' then
         i1 = torch.randn(ni, ih, iw, bs):cuda();
      else
         i1 = torch.randn(bs, ni, ih, iw):cuda();
      end
      collectgarbage()
      local o1 = mods[j]:forward(i1)
      cutorch.synchronize()
      collectgarbage()
      sys.tic()
      for t = 1,steps do
         o1 = mods[j]:updateOutput(i1)
      end
      cutorch.synchronize()
      timeOut[j] = sys.toc()/steps

      cutorch.synchronize()
      collectgarbage()
      sys.tic()
      for t = 1,steps do
         mods[j]:updateGradInput(i1, o1)
      end
      cutorch.synchronize()
      timeGradInput[j] = sys.toc()/steps

      cutorch.synchronize()
      collectgarbage()
      sys.tic()
      local ok = 1
      for t = 1,steps do
         ok = pcall(function() mods[j]:accGradParameters(i1, o1) end)
      end
      cutorch.synchronize()
      timeGradPara[j] = sys.toc()/steps

      collectgarbage()

   end

   local outMod, gradInputMod, gradParaMod
   local outMin = 100000
   local gradInputMin = 100000
   local gradParaMin = 100000
   for j=1,#mods do
      if timeOut[j] < outMin then
         outMin = timeOut[j]
         outMod = j
      end
      if timeGradInput[j] < gradParaMin then
         gradInputMin = timeGradInput[j]
         gradInputMod = j
      end
      if timeGradPara[j] < gradParaMin then
         gradParaMin = timeGradPara[j]
         gradParaMod = j
      end
   end
   self.playOutput = mods[outMod]
   self.playGradInput = mods[gradInputMod]
   self.playGradPara = mods[gradParaMod]
   local str = string.format(“%6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f %6.0f”, bs, ni, no, kw, kh, dw, dh, outMod, gradInputMod, gradParaMod)
   writefile(“data”, str)
end

function transposeInput(typename, input)
   if torch.typename(typename) == 'ccn2.SpatialConvolution' then
      input:transpose(1, 2);
      input:transpose(3, 2);
      input:transpose(3, 4);
   end
end

function SpatialConvolutionMetaFFT:updateOutput(input)
   -- print(self.playOutput)
   transposeInput(self.playOutput, input)
   return self.playOutput:updateOutput(input)
end

function SpatialConvolutionMetaFFT:updateGradInput(input, gradOutput)
   -- print(self.playGradInput)
   transposeInput(self.playGradInput, input)
   return self.playGradInput:updateGradInput(input, gradOutput)
end

function SpatialConvolutionMetaFFT:accGradParameters(input, gradOutput)
   -- print(self.playGradPara)
   transposeInput(self.playGradPara, input)
   return self.playGradPara:accGradParameters(input, gradOutput)
end

function SpatialConvolutionMetaFFT:reset()
   self.play:reset()
end

return metaHard
