require 'sys'
--require 'cunn'
require 'ccn2'
--require 'cudnn'
require 'fbcunn'
-- require 'nnbhwd'

-- print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

steps = 5 -- nb of steps in loop to average perf

runs = {
   
   {
      -- first layer
ni = 3,
no = 64,
kw = 11,
kh = 11,
iw = 128,
ih = 128,
bs = 64,
dw = 1,
dh = 1,
   }


}
--filters = { 128 }
--filters = { 32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512 }

--filters = { 128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512}

--for value, filter in ipairs(filters) do
for i,run in ipairs(runs) do
   -- params for run:
   local ni,bs,kw,kh,iw,ih,dw,dh,no = run.ni,run.bs,run.kw,run.kh,run.iw,run.ih,run.dw,run.dh,run.no
   --no = filter
   -- print('')
   -- print('CONFIG: input = ' .. ni..'x'..iw..'x'..ih..' * ker = ' .. ni..'x'..no..'x'..kw..'x'..kh .. ' (bs = '..bs..', stride = ' .. dw .. ')')
   local mods = {}
  -- mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh):cuda()
  -- mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
     mods[1] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
   -- mods[1] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()
   -- mods[4] = nn.SpatialConvolutionBHWD(ni,no,kw,kh,dw,dh):cuda()
   for j=1,#mods do   
      local tmf, tmbi, tmbg
      collectgarbage()
      if torch.typename(mods[j]) == 'ccn2.SpatialConvolution' then
         i1 = torch.randn(ni, ih, iw, bs):cuda();
      elseif torch.typename(mods[j]) == 'nn.SpatialConvolutionBHWD' then
         i1 = torch.randn(bs, ih, iw, ni):cuda();
      else
         i1 = torch.randn(bs, ni, ih, iw):cuda()
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
      tmf = sys.toc()/steps
      -- print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateOutput():', tmf*1000))

      cutorch.synchronize()
      collectgarbage()
      sys.tic()
      for t = 1,steps do
         mods[j]:updateGradInput(i1, o1)
      end
      cutorch.synchronize()
      tmbi = sys.toc()/steps
      -- print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateGradInput():', tmbi*1000))

      cutorch.synchronize()
      collectgarbage()
      sys.tic()
      local ok = 1
      for t = 1,steps do
         ok = pcall(function() mods[j]:accGradParameters(i1, o1) end)
      end
      cutorch.synchronize()
      tmbg = sys.toc()/steps
      if not ok then
         print(string.format("%-30s %25s %s", torch.typename(mods[j]), ':accGradParameters():', 'FAILED!'))
         tmbg = -0.001
      else
         -- print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':accGradParameters():', tmbg*1000))
      end
     print(string.format("%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f", bs, ni, no, ih, kh, dh, tmf * 1000, tmbi * 1000, tmbg*1000))--, output[4]))
      --foutput:flush()
   end
end
--end
--print('')
