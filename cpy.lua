require 'cutorch'
require 'nn'
require 'sys'
require 'cunn'
require 'ccn2'
require 'cudnn'
cudnn.benchmark = true -- run manual auto-tuner provided by cudnn
cudnn.verbose = false
require 'fbcunn'

kh = 3
kw = 3
ni = 3
no = 16
dw = 1
dh = 1
ih = 10
iw = 10
bs = 32

mods = {}
mods[1] = cudnn.SpatialConvolution(ni,no,kw,kh,dw,dh):cuda()
mods[2] = nn.SpatialConvolutionMM(ni,no,kw,kh,dw,dh):cuda()
mods[3] = ccn2.SpatialConvolution(ni,no,kw,dw,0,1,4):cuda()
mods[4] = nn.SpatialConvolutionCuFFT(ni,no,kw,kh,dw,dh):cuda()

transpose1 = nn.Transpose({1,2}, {2,3}, {3,4}):cuda()
transpose2 = nn.Transpose({4,1}, {4,2}, {4,3}):cuda()
transpose3 = nn.Transpose({1,2}):cuda()

i1 = torch.Tensor(bs, ni, ih, iw):cuda()
s = i1:storage()
for i = 1, s:size() do
	s[i] = i
end
i2 = transpose1:updateOutput(i1)

s = mods[2].weight:storage()
for i = 1, s:size() do
	s[i] = s:size() - i
end

mods[1].weight:copy(mods[2].weight)
mods[4].weight:copy(mods[2].weight)
mods[1].weight = transpose3:updateOutput(mods[2].weight)

s = mods[2].bias:storage()
for i = 1, s:size() do
	s[i] = 0--s:size() - i
end

mods[1].bias:copy(mods[2].bias)
mods[3].bias:copy(mods[2].bias)
mods[4].bias:copy(mods[2].bias)

o1 = mods[1]:forward(i1)
o2 = mods[2]:forward(i1)
o3 = mods[3]:forward(i2)
o4 = mods[4]:forward(i1)

o3 = transpose2:updateOutput(o3)

print(o1-o2)
print(o1-o3)
print(o1-o4)

-- here to observe output

s1 = o1:storage()
s2 = o2:storage()

for i = 1, s1:size() do
	s1[i] = 1
end

o2 = transpose2:updateGradInput(i2, o1)

gi1 = mods[1]:updateGradInput(i1, o1)
gi2 = mods[2]:updateGradInput(i1, o1)
gi3 = mods[3]:updateGradInput(i2, o2)
gi4 = mods[4]:updateGradInput(i1, o1)
-- here to observe gradinput
mods[1]:accGradParameters(i1, o1)
mods[2]:accGradParameters(i1, o1)
mods[3]:accGradParameters(i2, o2)
mods[4]:accGradParameters(i1, o1)

--[[
print(gi1)
print(gi2)
print(gi3)
print(gi4)
print(mods[1].weight)
print(mods[2].weight)
print(mods[3].weight)
print(mods[4].weight)
print(mods[1].gradWeight)
print(mods[2].gradWeight)
print(mods[3].gradWeight)
print(mods[4].gradWeight)
]]
-- here to observe gradWeight, mods[3] need to transpose