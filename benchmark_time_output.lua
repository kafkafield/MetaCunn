require 'sys'
require 'cunn'
require 'ccn2'
require 'cudnn'
cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
cudnn.verbose = false
require 'fbcunn'
-- require 'nnbhwd'

steps = 3 -- nb of steps in loop to average perf

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
   },
}

--filters = { 128 }
--filters = { 32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512 }

-- filters = {32,64,96,128,160,192,224,256,288,320,352,384,416,448,480,512}
batch = {32,
64,
96,
128,
160,
192,
224,
256,
288,
320,
352,
384,
416,
448,
480,
512,
}
input = {
32,
48,
64,
80,
96,
112,
128,
144,
160,
176,
192,
208,
224,
240,
256,
}
filters = {
32,
48,
64,
80,
96,
112,
128,
144,
160,
176,
192,
208,
224,
240,
256,
272,
288,
304,
320,
336,
352,
368,
384,
400,
416,
432,
448,
464,
480,
496,
512,
}
strides = {1,2,3,4,5,6,7,8,9,10,11,}
kernel = {
1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,
16,
17,
18,
19,
20,
21,
22,
23,
24,
25,
26,
27,
28,
29,
30,
31,
32,
33,
34,
35,
36,
37,
38,
39,
40,
41,
42,
43,
44,
45,
46,
47,
48,
49,
50,
}
channels = {
1,
3,
32,
48,
64,
80,
96,
112,
128,
144,
160,
176,
192,
208,
224,
240,
256,
272,
288,
304,
320,
336,
352,
368,
384,
400,
416,
432,
}
print('bs,ni,no,ih,kh,dh,cunn,ccn2,cudnn,fbfft')
for value, filter in ipairs(batch) do
for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   bs = filter
   -- print('bs,ni,no,ih,kh,dh,cunn,ccn2,cudnn,fbfft')
   local mods = {}
   local output = {}
   local gradInput = {}
   local gradPara = {}
   mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh):cuda()
   mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
   mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
   -- mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()
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
      output[j] = tmf*1000
      --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateOutput():', tmf*1000))

      cutorch.synchronize()
      collectgarbage()
      sys.tic()
      for t = 1,steps do
         mods[j]:updateGradInput(i1, o1)
      end
      cutorch.synchronize()
      tmbi = sys.toc()/steps
      gradInput[j] = tmf*1000
      --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateGradInput():', tmbi*1000))

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
         --print(string.format("%-30s %25s %s", torch.typename(mods[j]), ':accGradParameters():', 'FAILED!'))
      else
         gradPara[j] = tmbg * 1000
         --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':accGradParameters():', tmbg*1000))
      end
   end
   print(string.format("%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f", bs, ni, no, ih, kh, dh, output[1], output[2], output[3]))--, output[4]))
end
end

for value, filter in ipairs(filters) do
for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   no = filter
   -- print('bs,ni,no,ih,kh,dh,cunn,ccn2,cudnn,fbfft')
   local mods = {}
   local output = {}
   local greadInput = {}
   local gradPara = {}
   mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh):cuda()
   mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
   mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
   mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()
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
      output[j] = tmf*1000
      --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateOutput():', tmf*1000))

      cutorch.synchronize()
      collectgarbage()
      sys.tic()
      for t = 1,steps do
         mods[j]:updateGradInput(i1, o1)
      end
      cutorch.synchronize()
      tmbi = sys.toc()/steps
      gradInput[j] = tmf*1000
      --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateGradInput():', tmbi*1000))

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
         --print(string.format("%-30s %25s %s", torch.typename(mods[j]), ':accGradParameters():', 'FAILED!'))
      else
         gradPara[j] = tmbg * 1000
         --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':accGradParameters():', tmbg*1000))
      end
   end
   print(string.format("%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f", bs, ni, no, ih, kh, dh, output[1], output[2], output[3]))--, output[4]))
end
end

for value, filter in ipairs(channels) do
for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   ni = filter
   -- print('bs,ni,no,ih,kh,dh,cunn,ccn2,cudnn,fbfft')
   local mods = {}
   local output = {}
   local greadInput = {}
   local gradPara = {}
   mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh):cuda()
   mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
   mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
   mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()
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
      output[j] = tmf*1000
      --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateOutput():', tmf*1000))

      cutorch.synchronize()
      collectgarbage()
      sys.tic()
      for t = 1,steps do
         mods[j]:updateGradInput(i1, o1)
      end
      cutorch.synchronize()
      tmbi = sys.toc()/steps
      gradInput[j] = tmf*1000
      --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateGradInput():', tmbi*1000))

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
         --print(string.format("%-30s %25s %s", torch.typename(mods[j]), ':accGradParameters():', 'FAILED!'))
      else
         gradPara[j] = tmbg * 1000
         --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':accGradParameters():', tmbg*1000))
      end
   end
   print(string.format("%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f", bs, ni, no, ih, kh, dh, output[1], output[2], output[3]))--, output[4]))
end
end

for value, filter in ipairs(input) do
for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   ih = filter
   iw = filter
   -- print('bs,ni,no,ih,kh,dh,cunn,ccn2,cudnn,fbfft')
   local mods = {}
   local output = {}
   local greadInput = {}
   local gradPara = {}
   mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh):cuda()
   mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
   mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
   mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()
   -- mods[4] = nn.SpatialConvolutionBHWD(ni,no,kw,kh,dw,dh):cuda()
   for j=1,#mods do
      local tmf, tmbi, tmbg
      if torch.typename(mods[j]) == 'ccn2.SpatialConvolution' and (ih ~= 32 and ih ~= 64 and ih ~= 96 and ih ~= 128 and ih ~= 192 and ih ~= 256) then
         output[j] = nil
		 gradInput[j] = nil
		 gradPara[j] = nil
      else
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
         output[j] = tmf*1000
         --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateOutput():', tmf*1000))

         cutorch.synchronize()
         collectgarbage()
         sys.tic()
         for t = 1,steps do
            mods[j]:updateGradInput(i1, o1)
         end
         cutorch.synchronize()
         tmbi = sys.toc()/steps
         gradInput[j] = tmf*1000
         --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateGradInput():', tmbi*1000))

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
            --print(string.format("%-30s %25s %s", torch.typename(mods[j]), ':accGradParameters():', 'FAILED!'))
         else
            gradPara[j] = tmbg * 1000
            --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':accGradParameters():', tmbg*1000))
         end
      end
   end
   print(string.format("%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f", bs, ni, no, ih, kh, dh, output[1], output[2], output[3]))--, output[4]))
end
end

for value, filter in ipairs(kernel) do
for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   kh = filter
   kw = filter
   -- print('bs,ni,no,ih,kh,dh,cunn,ccn2,cudnn,fbfft')
   local mods = {}
   local output = {}
   local greadInput = {}
   local gradPara = {}
   mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh):cuda()
   mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
   mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
   mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()
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
      output[j] = tmf*1000
      --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateOutput():', tmf*1000))

      cutorch.synchronize()
      collectgarbage()
      sys.tic()
      for t = 1,steps do
         mods[j]:updateGradInput(i1, o1)
      end
      cutorch.synchronize()
      tmbi = sys.toc()/steps
      gradInput[j] = tmf*1000
      --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateGradInput():', tmbi*1000))

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
         --print(string.format("%-30s %25s %s", torch.typename(mods[j]), ':accGradParameters():', 'FAILED!'))
      else
         gradPara[j] = tmbg * 1000
         --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':accGradParameters():', tmbg*1000))
      end
   end
   print(string.format("%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f", bs, ni, no, ih, kh, dh, output[1], output[2], output[3]))--, output[4]))
end
end

for value, filter in ipairs(strides) do
for i,run in ipairs(runs) do
   -- params for run:
   local ni,no,kw,kh,bs,iw,ih,dw,dh = run.ni,run.no,run.kw,run.kh,run.bs,run.iw,run.ih,run.dw,run.dh
   dw = filter
   dh = filter
   -- print('bs,ni,no,ih,kh,dh,cunn,ccn2,cudnn,fbfft')
   local mods = {}
   local output = {}
   local greadInput = {}
   local gradPara = {}
   mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh):cuda()
   mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
   mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
   mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()
   -- mods[4] = nn.SpatialConvolutionBHWD(ni,no,kw,kh,dw,dh):cuda()
   for j=1,#mods do
      local tmf, tmbi, tmbg
      if torch.typename(mods[j]) == 'nn.SpatialConvolutionCuFFT' and dh > 1 then
         output[j] = nil
		 gradInput[j] = nil
		 gradPara[j] = nil
      else
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
         output[j] = tmf*1000
         --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateOutput():', tmf*1000))

         cutorch.synchronize()
         collectgarbage()
         sys.tic()
         for t = 1,steps do
            mods[j]:updateGradInput(i1, o1)
         end
         cutorch.synchronize()
         tmbi = sys.toc()/steps
         gradInput[j] = tmf*1000
         --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':updateGradInput():', tmbi*1000))

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
            --print(string.format("%-30s %25s %s", torch.typename(mods[j]), ':accGradParameters():', 'FAILED!'))
         else
            gradPara[j] = tmbg * 1000
            --print(string.format("%-30s %25s %10.2f", torch.typename(mods[j]), ':accGradParameters():', tmbg*1000))
         end
      end
   end
   print(string.format("%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f,%10.2f", bs, ni, no, ih, kh, dh, output[1], output[2], output[3]))--, output[4]))
end
end

print('')
