-- protocol: batchsize, inputPlane, outputPlane, kernelWidth, kernelHeight, inputWidth, inputHeight, strideWidth, strideHeight, Output, Input, Gradacc. 
metaHard = {}

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


function writefile(filename, info)
	local wfile=io.open(filename, "a")
	assert(wfile)
	wfile:write(info)
	wfile:close()
end

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
   mods = {}
   mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh) 
   mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh)
   mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4)
   mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh)
   outR, gradInputR, gradParaR = metahard.loadmap(bs,ni,no,kw,kh,iw,ih,dw,dh)

   local outMod, gradInputMod, gradParaMod
   if outR > 0 then
      outMod = outR
      gradInputMod = gradInputR
      gradParaMod = gradParaR
      if torch.typename(mods[gradInputMod]) == 'nn.SpatialConvolutionCuFFT' or torch.typename(mods[gradParaMod]) == 'nn.SpatialConvolutionCuFFT' then
         mods[4]:cuda()
         i1 = torch.randn(bs, ni, ih, iw):cuda()
         collectgarbage()
         mods[4]:forward(i1)
         cutorch.synchronize()
         collectgarbage()
      end
   else
      print("Need choice !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      timeOut = {}
      timeGradInput = {}
      timeGradPara = {}
      memOut = {}
      memGradInput = {}
      memGradPara = {}
      local steps = 1
      
      for j=1,#mods do
         mods[j]:cuda()
         collectgarbage()
         if torch.typename(mods[j]) == 'ccn2.SpatialConvolution' then
            i1 = torch.randn(ni, ih, iw, bs):cuda();
         else
            i1 = torch.randn(bs, ni, ih, iw):cuda();
         end
         collectgarbage()
         local o1 = mods[j]:forward(i1)
         cutorch.synchronize()

         os.execute('nvidia-smi --query-gpu=memory.used --format=csv -lms 1 -f ./heihei &')
         collectgarbage()
         sys.tic()
         for t = 1,steps do
            o1 = mods[j]:updateOutput(i1)
         end
         cutorch.synchronize()
         timeOut[j] = sys.toc()/steps
         os.execute('pgrep nvidia-smi | xargs kill -s 9')
         memOut[j] = metahard.getMaxMemory()
         -- print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), 'Memory :updateOutput():', memOut[j]))

         os.execute('nvidia-smi --query-gpu=memory.used --format=csv -lms 1 -f ./heihei &')
         cutorch.synchronize()
         collectgarbage()
         sys.tic()
         for t = 1,steps do
            mods[j]:updateGradInput(i1, o1)
         end
         timeGradInput[j] = sys.toc()/steps
         cutorch.synchronize()
         os.execute('pgrep nvidia-smi | xargs kill -s 9')
         memGradInput[j] = metahard.getMaxMemory()
         -- print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), 'Memory :updateGradInput():', memGradInput[j]))

         os.execute('nvidia-smi --query-gpu=memory.used --format=csv -lms 1 -f ./heihei &')
         cutorch.synchronize()
         collectgarbage()
         sys.tic()
         local ok = 1
         for t = 1,steps do
            ok = pcall(function() mods[j]:accGradParameters(i1, o1) end)
         end
         cutorch.synchronize()
         timeGradPara[j] = sys.toc()/steps
         os.execute('pgrep nvidia-smi | xargs kill -s 9')
         memGradPara[j] = metahard.getMaxMemory()
         -- print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), 'Memory :accGradParameters():', memGradPara[j]))

         collectgarbage()
         mod[j]:float()
      end


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
      local str = string.format("%d %d %d %d %d %d %d %d %d %d %d %d\n", bs, ni, no, kw, kh, iw, ih, dw, dh, outMod, gradInputMod, gradParaMod)
      writefile("data", str)
   end
   self.playOutput = mods[outMod]
   self.playGradInput = mods[gradInputMod]
   self.playGradPara = mods[gradParaMod]
end

function transposeInput(typename, input)
   if torch.typename(typename) == 'ccn2.SpatialConvolution' then
      input:transpose(1, 2);
      input:transpose(3, 2);
      input:transpose(3, 4);
   end
end

function SpatialConvolutionMetaHard:updateOutput(input)
   -- print(self.playOutput)
   transposeInput(self.playOutput, input)
   return self.playOutput:updateOutput(input)
end

function SpatialConvolutionMetaHard:updateGradInput(input, gradOutput)
   -- print(self.playGradInput)
   transposeInput(self.playGradInput, input)
   return self.playGradInput:updateGradInput(input, gradOutput)
end

function SpatialConvolutionMetaHard:accGradParameters(input, gradOutput)
   -- print(self.playGradPara)
   transposeInput(self.playGradPara, input)
   return self.playGradPara:accGradParameters(input, gradOutput)
end

function SpatialConvolutionMetaHard:reset()
   self.play:reset()
end

return metaHard
