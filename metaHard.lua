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

transpose1 = nn.Transpose({1,2}, {2,3}, {3,4}):cuda()
transpose2 = nn.Transpose({4,1},{4,2},{4,3}):cuda()
transposeKernel = nn.Transpose({1,2}):cuda()

function SpatialConvolutionMetaHard:__init(nInputPlane, nOutputPlane,
                                        kW, kH, dW, dH, paW, paH)
   parent.__init(self)
   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.paW = paW or 0
   self.paH = paH or 0

   self.configed = false
   self.testConfig = 0

   -- kernel
   self.weight = torch.Tensor(nOutputPlane, nInputPlane * kH * kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane * kH * kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self:reset()
end

function SpatialConvolutionMetaHard:config(size)

   -- for test
   self.testConfig = self.testConfig + 1

   ni = self.nInputPlane
   no = self.nOutputPlane
   kw = self.kW
   kh = self.kH
   dw = self.dW
   dh = self.dH
   paW = self.paW
   paH = self.paH
   ih = size[3]
   iw = size[4]
   bs = size[1]
   mods = {}
   mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh) 
   mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh)
   mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4)
   if dh == 1 then
      mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh)
   end
   outR, gradInputR, gradParaR = metahard.loadmap(bs,ni,no,kw,kh,iw,ih,dw,dh)

   local outMod, gradInputMod, gradParaMod
   if outR > 0 then
      outMod = outR
      gradInputMod = gradInputR
      gradParaMod = gradParaR
   else
      -- print("Need choice !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      timeOut = {}
      timeGradInput = {}
      timeGradPara = {}
      memOut = {}
      memGradInput = {}
      memGradPara = {}
      local steps = 1
      
      for j=1,#mods do
         collectgarbage()
         if (torch.typename(mods[j]) == 'nn.SpatialConvolutionCuFFT' and dh > 1) or (torch.typename(mods[j]) == 'ccn2.SpatialConvolution' and (no ~= 32 and no ~= 64 and no ~= 96 and no ~= 128 and no ~= 192 and no ~= 256)) then
            timeOut[j] = 100000
            timeGradInput[j] = 100000
            timeGradPara[j] = 100000
         else
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
            if not ok then
               timeGradPara[j] = 1000000
            end
            os.execute('pgrep nvidia-smi | xargs kill -s 9')
            memGradPara[j] = metahard.getMaxMemory()
            -- print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), 'Memory :accGradParameters():', memGradPara[j]))

            collectgarbage()
            mods[j]:float()
         end
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
      --local log = string.format("%10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n", timeOut[1]*1000, timeOut[2]*1000, timeOut[3]*1000, timeOut[4]*1000, timeGradInput[1]*1000, 
      --   timeGradInput[2]*1000, timeGradInput[3]*1000, timeGradInput[4]*1000, timeGradPara[1]*1000, timeGradPara[2]*1000, timeGradPara[3]*1000, timeGradPara[4]*1000)
      --writefile('metahardtest', log)
      local str = string.format("%d %d %d %d %d %d %d %d %d %d %d %d\n", bs, ni, no, kw, kh, iw, ih, dw, dh, outMod, gradInputMod, gradParaMod)
      writefile("data", str)
   end
   self.playOutput = mods[outMod]:cuda()
   self.playGradInput = mods[gradInputMod]:cuda()
   self.playGradPara = mods[gradParaMod]:cuda()

   -- forward to initialize
   i1 = torch.randn(bs, ni, ih, iw):cuda()
   i2 = torch.randn(ni, ih, iw, bs):cuda()
   collectgarbage()
   if torch.typename(self.playOutput) ~= 'ccn2.SpatialConvolution' then
      self.playOutput:forward(i1)
   else
      self.playOutput:forward(i2)
   end
   cutorch.synchronize()
   collectgarbage()
   if torch.typename(self.playGradInput) ~= 'ccn2.SpatialConvolution' then
      self.playGradInput:forward(i1)
   else
      self.playGradInput:forward(i2)
   end
   cutorch.synchronize()
   collectgarbage()
   if torch.typename(self.playGradPara) ~= 'ccn2.SpatialConvolution' then
      self.playGradPara:forward(i1)
   else
      self.playGradPara:forward(i2)
   end
   cutorch.synchronize()
   collectgarbage()

   self.configed = true
   --print(self.playOutput)
   --print(self.playGradInput)
   --print(self.playGradPara)
end

function SpatialConvolutionMetaHard:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function transposeInput(typename, input)
   if torch.typename(typename) == 'ccn2.SpatialConvolution' then
      input:transpose(1, 2);
      input:transpose(3, 2);
      input:transpose(3, 4);
   end
end

function SpatialConvolutionMetaHard:copykernelMtoI(inplem)
   if torch.typename(typename) == 'ccn2.SpatialConvolution' then
      inplem.gradWeight = transposeKernel:updateOutput(self.gradWeight)
      inplem.gradBias:copy(self.gradBias)
   else
      inplem.gradWeight:copy(self.gradWeight)
      inplem.gradBias:copy(self.gradBias)
   end
end

function SpatialConvolutionMetaHard:copykernelItoM(inplem)
   if torch.typename(typename) == 'ccn2.SpatialConvolution' then
      self.gradWeight = transposeKernel:updateOutput(inplem.gradWeight)
      self.gradBias:copy(inplem.gradBias)
   else
      self.gradWeight:copy(inplem.gradWeight)
      self.gradBias:copy(inplem.gradBias)
   end
end

function SpatialConvolutionMetaHard:updateOutput(input)
   if not self.configed then
      self:config(input:size()) 
   end
   if torch.typename(self.playOutput) == 'ccn2.SpatialConvolution' then
      input2 = transpose1:updateOutput(input)
      out = self.playOutput:updateOutput(input2)
      --print(input2:size()) 
      return transpose2:updateOutput(out)
   else
      --print(input:size())
      return self.playOutput:updateOutput(input)
   end
end

function SpatialConvolutionMetaHard:updateGradInput(input, gradOutput)
   if torch.typename(self.playGradInput) == 'ccn2.SpatialConvolution' then
      input2 = transpose1:updateOutput(input)
      --print(input2:size())
      gradOutput2 = transpose2:updateGradInput(input2, gradOutput)
      gradOutput3 = self.playGradInput:updateGradInput(input2, gradOutput2)
      gradOutput4 = transpose1:updateGradInput(input2, gradOutput3)
      --print(gradOutput:size())
      --print(gradOutput3:size())
      return gradOutput4
   else
      return self.playGradInput:updateGradInput(input, gradOutput)
   end
end

function SpatialConvolutionMetaHard:accGradParameters(input, gradOutput)
   if torch.typename(self.playGradPara) == 'ccn2.SpatialConvolution' then
      input2 = transpose1:updateOutput(input)
      gradOutput2 = transpose2:updateGradInput(input2, gradOutput)
      self.playGradPara:accGradParameters(input2, gradOutput2)
   else
      self.playGradPara:accGradParameters(input, gradOutput)
   end
   self:copykernelItoM(self.playGradPara)
   self:copykernelMtoI(self.playOutput)
   self:copykernelMtoI(self.playGradInput)
end

return metaHard
