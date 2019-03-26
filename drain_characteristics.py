import smu
import numpy as np

s = smu.smu()

pmos = False

Vt = .56
if pmos:
    Vt = 4.34

v = np.linspace(0,5,1000)

if pmos:
    v = 5 - v

vg = [Vt-.1,Vt,5]

if pmos:
    vg = [Vt+.1,Vt,0]

#for Vg in vg:
for Vg in [0.46] :
    f = open("data/exp3_{!s}mos_{!s}.csv".format('p' if pmos else 'n',Vg),'w')
    f.write("Vd, I\n")
    
    s.set_voltage(1,0)
    s.autorange(1)
    s.set_voltage(2,0)
    s.autorange(2)
    for val in v:
        s.set_voltage(1,Vg)
        
        s.autorange(1)
        s.set_voltage(2,val)
        #if (val < 2):
        s.autorange(2)
        f.write('{!s},{!s}\n'.format(val,s.get_current(2)))

s.set_current(1,0)
f.close()
