require 'image'
require 'nn'

---------------------------
--       Constants       --
---------------------------

TrainningImagesRouteF = '/home/mcolula/colorferet/cropped/F64x64/'
TrainningImagesRouteM = '/home/mcolula/colorferet/cropped/M64x64/'
NumberOfOut  =     1
NumberOfIns  =  4096
LearningRate = 0.001


---------------------------
--    Helper Funtions    --
---------------------------


function rgb2Gray(img)

  local h = img:size()[2]
  local w = img:size()[3]

  local grayImage = torch.Tensor(w, h):zero()

  for i = 1, w do
    for j = 1, h do
      if img:size()[1] > 1 then
        grayImage[i][j] = (img[1][i][j] + img[2][i][j] + img[3][i][j]) / 3
      else
        grayImage[i][j] =  img[1][i][j]
      end
    end
  end

  return grayImage

end

function loadImagesRgb(dir)
  idx = 0
  ims = {}
  for name in paths.files(dir) do
    path = dir .. name
    if string.match(path, '.*png') then
      ims[idx] = image.loadPNG(path)
      idx = idx + 1
    end
  end
  return ims
end


function loadImagesGray(dir)
  ims = {}
  for idx, img in pairs(loadImagesRgb(dir)) do
    ims[idx] = rgb2Gray(img)
  end
  return ims
end

function xyTo1D(img)

  local w = img:size()[1]
  local h = img:size()[2]
  local flat = torch.Tensor(w * h):zero()

  for i = 1, w do
    for j = 1, h do
      flat[(i - 1) * h + j] = img[i][j]
    end
  end

  return flat

end


---------------------------
--     Neural Network    --
---------------------------

genderClassifier = nn.Sequential()

genderClassifier:add(nn.Linear(NumberOfIns, NumberOfOut))

criterion = nn.MSECriterion()


---------------------------
--       Trainning       --
---------------------------

for i, img in pairs(loadImagesGray(TrainningImagesRouteF)) do

  local out = torch.Tensor(NumberOfOut)
  local ins = xyTo1D(img)

  out[1] = 1

  criterion:forward(genderClassifier:forward(ins), out)

  genderClassifier:zeroGradParameters()

  genderClassifier:backward(ins, criterion:backward(genderClassifier.output, out))

  genderClassifier:updateParameters(LearningRate)

end

for i, img in pairs(loadImagesGray(TrainningImagesRouteM)) do

  local out = torch.Tensor(NumberOfOut)
  local ins = xyTo1D(img)

  out[1] = 0

  criterion:forward(genderClassifier:forward(ins), out)

  genderClassifier:zeroGradParameters()

  genderClassifier:backward(ins, criterion:backward(genderClassifier.output, out))

  genderClassifier:updateParameters(LearningRate)

end

print('Female testing')
for i, img in pairs(loadImagesGray(TrainningImagesRouteF)) do

  local ins = xyTo1D(img)

  print(genderClassifier:forward(ins), out)

end

print('Male testing')
for i, img in pairs(loadImagesGray(TrainningImagesRouteM)) do

  local ins = xyTo1D(img)

  print(genderClassifier:forward(ins), out)

end
