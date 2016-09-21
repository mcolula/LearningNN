require 'nn'

------------------------------
--        Constantes        --
------------------------------

D =     2 -- Demensionalidad
K =     1 -- Número de class
N =   100 -- Número de datos
H =   100 -- Hidden    units
R = 0.001 -- Learning   rate

------------------------------
--       Gen. de datos      --
------------------------------


-- Creación de los datos de entrenamiento --

-- usamos una funcion cuadrática para que queden parábolas --

f = {}
f[1] = torch.linspace(-1, 1, N) + torch.randn(N) * 0.05
f[2] = torch.pow(f[1], 2)       + torch.randn(N) * 0.05


-- esta parábola se desplaza para que el problema sea separable --
g = {}
g[1] = torch.linspace(-1, 1, N) + torch.randn(N) * 0.05
g[2] = torch.pow(g[1], 2) * -1  + torch.randn(N) * 0.05 + torch.randn(N):fill(1.5)
g[1] = g[1] + torch.randn(N):fill(1)


-- Usamos randn para que se vean pro los puntos --


------------------------------
--         Topología        --
------------------------------

model = nn.Sequential()
model:add(nn.Linear(D, H))
model:add(nn.ReLU())
model:add(nn.Linear(H, K))

criterion = nn.MSECriterion()

------------------------------
--         Trainning        --
------------------------------

for train = 1, 1000 do

  for i = 1, N do

    local ins = torch.randn( 2)
    local out = torch.Tensor(1)

    ins[1] = f[1][i]
    ins[2] = f[2][i]
    out[1] = 1       -- Clase A se representa por un uno

    criterion:forward(model:forward(ins), out)

    model:zeroGradParameters()
    model:backward(ins, criterion:backward(model.output, out))
    model:updateParameters(R)

  end

  for i = 1, N do

    local ins = torch.randn( 2)
    local out = torch.Tensor(1)

    ins[1] = g[1][i]
    ins[2] = g[2][i]
    out[1] = 0       -- Clase B se representa por un cero

    criterion:forward(model:forward(ins), out)

    model:zeroGradParameters()
    model:backward(ins, criterion:backward(model.output, out))
    model:updateParameters(R)

  end

  --print('Error: ' .. criterion.output)

end


------------------------------
--         Trainning        --
------------------------------


sdx = 0
tdx = 0
s  = {} -- Puntos clase A
t  = {} -- Puntos clase B

for x = -2, 2, 0.01 do
  for y =  0, 2, 0.01 do
    model:forward(torch.Tensor{x, y})
    if model.output[1] >= 0.5 then
      sdx = sdx + 1
      s[sdx] = torch.Tensor{x, y}
    else
      tdx = tdx + 1
      t[tdx] = torch.Tensor{x, y}
    end
  end
end


-- Se tienen que imprimir así para que gnuplot funcione --

for i = 1, sdx do
  print(s[i][1] .. ' ' .. s[i][2])
end

print('\n\n\n\n')

for i = 1, tdx do
  print(t[i][1] .. ' ' .. t[i][2])
end
