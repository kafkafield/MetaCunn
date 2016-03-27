-- protocol: batchsize, inputPlane, outputPlane, kernelWidth, kernelHeight, inputWidth, inputHeight, strideWidth, strideHeight, Output, Input, Gradacc. 
metaRegression = {}

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

regressall = {

   {
      intercept   = -1.366e+02,
      bs          = 7.193e-01,
      ih2         = 2.876e-03,
      ni          = 1.230e+01,
      no          = 3.733e-01,
      nofix64     = 4.667e-01,
      kh          = 1.389e+01,
      khlog       = -6.676e+01,
      dhm2        = 2.751e+01,
   },
   {
      intercept   = -3.100e+02,
      bs          = 8.342e-01,
      ih2         = 2.967e-03,
      no          = 2.358e-01,
      ni          = 1.589e+01,
      dhm2        = 5.808e+01,
      kh          = 1.325e+01,
      kh3         = 1.680e-03,
   },
   {
      intercept   =-2.695e+02,
      bs          = 1.202e+00,
      ih2         = 4.637e-03,
      no3         = 2.076e-06,
      nofix64     = 1.361e+00,
      kh3         = 1.489e-03,
      kh          = 7.377e+00,
      ni          = 1.405e+01,
      dhm2        = 9.330e+01,
   },
   {
      intercept   =-1.822e+02,
      bs          = 6.535e-01,
      ih2         = 2.813e-03,
      dhm2        = 4.156e+01,
      kh          = 8.353e+00,
      khlog       =-2.228e+01,
      ni          = 1.146e+01,
      no          = 3.375e-01,
      nofix64     = 2.747e-01,
   },
   {
      intercept   =-2.403e+02,
      bs          = 8.502e-01,
      ih2         = 3.638e-03,
      ni          = 1.729e+01,
      no          = 2.587e-01,
      kh          = 1.233e+01,
      khlog       =-3.231e+01,
      dhm2        = 5.611e+01,
   },
   {
      intercept   =-2.775e+02,
      bs          = 8.797e-01,
      ih2         = 4.020e-03,
      no3         = 3.520e-06,
      nofix128    = 6.221e-01,
      khlog       =-2.588e+01,
      kh          = 1.141e+01,
      ni          = 1.680e+01,
      dhm2        = 8.183e+01,
   },
   {
      intercept   =-2.075e+02,
      bs          = 9.337e-01,
      bsfix128    =-2.305e-01,
      bsfix32     =-3.248e-01,
      ih2         = 2.604e-03,
      no          = 6.053e-01,
      khlog       =-2.411e+01,
      kh          = 8.743e+00,
      dhm2        = 3.884e+01,
      ni          = 1.635e+01,
   },
   {
      intercept   =-3.985e+02,
      bs          = 1.500e+00,
      bsfix32     =-5.029e-01,
      ni          = 1.721e+01,
      no          = 1.113e+00,
      kh          = 1.225e+01,
      dhm2        = 7.288e+01,
      ih2         = 4.706e-03,
   },
   {
      intercept   =-1.918e+02,
      bs          = 6.654e-01,
      ih2         = 2.898e-03,
      no          = 6.745e-01,
      khlog       =-3.465e+01,
      kh          = 1.041e+01,
      dhm2        = 4.408e+01,
      ni          = 9.171e+00,
   },
   {
      intercept   =-4.476e+01,
      bs          = 3.366e-01,
      ihl1442     = 1.422e-03,
      ihh1442     =-3.626e-04,
      ihh144      = 2.636e-01,
      no          = 3.376e-01,
      kh          =-6.783e-02,
      ni          = 7.343e-01,
   },
   {
      intercept   =-4.340e+01,
      bs          = 3.291e-01,
      ihl1442     = 1.304e-03,
      ihh1442     =-8.183e-04,
      ihh144      = 3.313e-01,
      no          = 3.220e-01,
      kh          = 3.081e-03,
      ni          = 7.411e-01,
   },
   {
      intercept   =-4.888e+01,
      bs          = 3.757e-01,
      ihl1442     = 1.516e-03,
      ihh1442     =-5.747e-04,
      ihh144      = 3.284e-01,
      no          = 3.685e-01,
      kh          =-4.794e-02,
      ni          = 7.402e-01,
   },
   {
      intercept                              =-78.017823,
      bs                                     = 0.380242,
      bsl160                     =-0.103628,
      ih                                     = 0.449422,
      ih64144  =-0.470662,
      ihh128             =-0.281939,
      ihl80                      =-0.194549,
      no                                     = 0.257135,
      kh                                     = 2.070089,
      khh6                 =-2.115436,
      dhm1                                   =17.363520,
      ni                                     =11.358561,
      nil288                     =-10.342842,
   },
   {
      intercept                                  =-243.75475,
      bs                                         = 0.82888,
      ih                                         = 0.72452,
      no                                         = 0.24839,
      kh                                         = 6.64343,
      khl15                  =-6.51113,
      dhm2                                       =57.69283,
      ni                                         =16.08099,
      nil144                       =-13.05797,
      ni144224   = 2.17940,
      ni208272    =-8.60979,
      ni256320    = 3.57849,
      nih320                 =-15.37197,
   },
   {
      intercept                           =-1.144e+02,
      bs                                    = 4.273e-01,
      bsh512                   = 2.581e-01,
      ih                                    = 8.356e-01,
      ih64144 =-6.776e-01,
      ihh144            =-5.584e-01,
      no                                    = 3.900e-01,
      noh496                   = 1.044e-01,
      kh                                    =-1.489e-01,
      khl4                    =-3.554e+00,
      dhm2                                  = 2.744e+01,
      nil384                    =-1.539e+01,
      ni                                    = 1.610e+01,
   },

}

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
      --local steps = 1

      -- regression model
      kh3 = kh^3
      ih2 = ih^2
      nofix128 = math.ceil(no/128)*128-no
      khlog = math.log(kh)
      dhm2 = 1/dh^2
      kh2 = (1+(-1)^(math.ceil((kh+2)%4/4)))/2*kh
      kh4 = (1+(-1)^(math.ceil((kh)%4/4)))/2*kh
      no3 = no^3
      nofix64 = math.ceil(no/64)*64 - no
      bsfix32 = (1+(-1)^(math.ceil((bs)%64/64)))/2*bs
      bsfix128 = (1+(-1)^(math.ceil((bs)%128/128)))/2*bs
      ihl144 = -ih*math.floor((ih-144)/256)
      ihh144 = -ih*math.floor((143-ih)/256)
      dhm1 = 1/dh
      ihl1442 = ihl144^2
      ihh1442 = ihh144^2
      bsl160 = (bs < 160) and bs or 0
      ih64144 = (ih > 64 and ih < 144) and ih or 0
      ihh128 = (ih > 128) and ih or 0
      ihl80 = (ih < 80) and ih or 0
      khh6 = (kh > 6) and kh or 0
      nil288 = (ni < 288) and ni or 0
      khl15 = (kh < 15) and kh or 0
      nil144 = (ni < 144) and ni or 0
      ni144224 = (ni > 144 and ni < 224) and ni or 0
      ni208272 = (ni > 208 and ni < 272) and ni or 0
      ni256320 = (ni > 256 and ni < 320) and ni or 0
      nih320 = (ni > 320) and ni or 0
      bsh512 = (bs >= 512) and bs or 0
      noh496 = (no > 496) and no or 0
      khl4 = (kh < 4) and kh or 0
      nil384 = (ni < 384) and ni or 0


      timeOut[2] = regressall[1].intercept + regressall[1].bs*bs + regressall[1].ih2 * ih2 + regressall[1].ni * ni + regressall[1].no * no + 
         regressall[1].nofix64 * nofix64 + regressall[1].khlog * khlog + regressall[1].kh * kh + regressall[1].dhm2 * dhm2   
      timeGradInput[2] = regressall[2].intercept + regressall[2].bs*bs + regressall[2].ih2 * ih2 + regressall[2].ni * ni + regressall[2].no * no + 
         regressall[2].kh3 * kh3 + regressall[2].kh * kh + regressall[2].dhm2 * dhm2
      timeGradPara[2] = regressall[3].intercept + regressall[3].bs*bs + regressall[3].ih2 * ih2 + regressall[3].ni * ni + regressall[3].no3 * no3 + 
         regressall[3].nofix64 * nofix64 + regressall[3].kh3 * kh3 + regressall[3].kh * kh + regressall[3].dhm2 * dhm2

--[[      timeOut[1] = regressall[4].intercept + regressall[4].bs*bs + regressall[4].ih2 * ih2 + regressall[4].ni * ni + regressall[4].no * no + 
         regressall[4].nofix64 * nofix64 + regressall[4].khlog * khlog + regressall[4].kh * kh + regressall[4].dhm2 * dhm2
      timeGradInput[1] = regressall[5].intercept + regressall[5].bs*bs + regressall[5].ih2 * ih2 + regressall[5].ni * ni + regressall[5].no * no + 
         regressall[5].khlog * khlog + regressall[5].kh * kh + regressall[5].dhm2 * dhm2
      timeGradPara[1] = regressall[6].intercept + regressall[6].bs*bs + regressall[6].ih2 * ih2 + regressall[6].ni * ni + regressall[6].no3 * no3 + 
         regressall[6].nofix128 * nofix128 + regressall[6].khlog * khlog + regressall[6].kh * kh + regressall[6].dhm2 * dhm2]]

      timeOut[1] = regressall[13].intercept + regressall[13].bs*bs + regressall[13].bsl160*bsl160 + regressall[13].ih * ih + regressall[13].ih64144 * ih64144 + regressall[13].ihh128 * ihh128 + 
         regressall[13].ihl80 * ihl80 + regressall[13].no * no + regressall[13].kh * kh + regressall[13].khh6 * khh6 + regressall[13].dhm1 * dhm1 + regressall[13].ni * ni + regressall[13].nil288 * nil288
      timeGradInput[1] = regressall[14].intercept + regressall[14].bs*bs + regressall[14].ih * ih + regressall[14].no * no + 
         regressall[14].khl15 * khl15 + regressall[14].kh * kh + regressall[14].dhm2 * dhm2 + regressall[14].ni * ni + regressall[14].nil144 * nil144 + regressall[14].ni144224 * ni144224 + 
         regressall[14].ni208272 * ni208272 + regressall[14].ni256320 * ni256320 + regressall[14].nih320 * nih320
      timeGradPara[1] = regressall[15].intercept + regressall[15].bs*bs + regressall[15].bsh512*bsh512 + regressall[15].ih * ih + regressall[15].ih64144 * ih64144 + regressall[15].ihh144 * ihh144 + 
         regressall[15].no * no + 
         regressall[15].noh496 * noh496 + regressall[15].khl4 * khl4 + regressall[15].kh * kh + regressall[15].dhm2 * dhm2 + regressall[15].ni * ni + regressall[15].nil384 * nil384

      timeOut[3] = regressall[7].intercept + regressall[7].bs*bs + regressall[7].bsfix128*bsfix128+ regressall[7].bsfix32*bsfix32 + regressall[7].ih2 * ih2 + 
         regressall[7].ni * ni + regressall[7].no * no + regressall[7].khlog * khlog + regressall[7].kh * kh + regressall[7].dhm2 * dhm2
      timeGradInput[3] = regressall[8].intercept + regressall[8].bs*bs + regressall[8].bsfix32*bsfix32 + regressall[8].ih2 * ih2 + 
         regressall[8].ni * ni + regressall[8].no * no + regressall[8].kh * kh + regressall[8].dhm2 * dhm2
      timeGradPara[3] = regressall[9].intercept + regressall[9].bs*bs + regressall[9].ih2 * ih2 + regressall[9].khlog * khlog + 
         regressall[9].ni * ni + regressall[9].no * no + regressall[9].kh * kh + regressall[9].dhm2 * dhm2
      timeOut[4] = regressall[10].intercept + regressall[10].bs*bs + regressall[10].ihl1442 * ihl1442 + 
         regressall[10].ihh1442 * ihh1442 +regressall[10].ihh144 * ihh144 + regressall[10].ni * ni + regressall[10].no * no + regressall[10].kh * kh
      timeGradInput[4] = regressall[11].intercept + regressall[11].bs*bs + regressall[11].ihl1442 * ihl1442 + 
         regressall[11].ihh1442 * ihh1442 +regressall[11].ihh144 * ihh144 + regressall[11].ni * ni + regressall[11].no * no + regressall[11].kh * kh
      timeGradPara[4] = regressall[12].intercept + regressall[12].bs*bs  + regressall[12].ihl1442 * ihl1442 + 
         regressall[12].ihh1442 * ihh1442 +regressall[12].ihh144 * ihh144 + regressall[12].ni * ni + regressall[12].no * no + regressall[12].kh * kh
      if (dh > 1) then
         timeOut[4] = 100000
         timeGradInput[4] = 100000
         timeGradPara[4] = 100000
      end
      if (no ~= 32 and no ~= 64 and no ~= 96 and no ~= 128 and no ~= 192 and no ~= 256) then
         timeOut[3] = 100000
         timeGradInput[3] = 100000
         timeGradPara[3] = 100000
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

   -- kernel
   self.weight = torch.Tensor(nOutputPlane, nInputPlane * kH * kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane * kH * kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self:reset()
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
   if torch.typename(self.playOutput) == 'ccn2.SpatialConvolution' then
      input2 = transpose1:updateOutput(input)
      out = self.playOutput:updateOutput(input2)
      --print(input2:size())
      return transpose2:updateOutput(out)
   else
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
--[[
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
]]
return metaRegression
