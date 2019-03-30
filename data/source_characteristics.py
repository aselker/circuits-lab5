import smu
import numpy as np

s = smu.smu()

pmos = True

v = np.linspace(0,5,1000)

if pmos:
    v = 5-v

f = open("data/exp2_pmos.csv",'w')
f.write("Vs, I\n")

s.set_voltage(1,0 if pmos else 5)
s.autorange(1)
s.set_voltage(2,0)
s.autorange(2)

for val in v:
    s.set_voltage(1,0 if pmos else 5)
    s.autorange(1)
    s.set_voltage(2,val)
    s.autorange(2)
    f.write('{!s},{!s}\n'.format(val,s.get_current(2)))

s.set_current(1,0)
f.close()
