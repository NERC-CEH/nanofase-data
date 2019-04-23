from pint import UnitRegistry
import numpy as np

ureg = UnitRegistry()
from_units = 'mm / day'
to_units = 'm / day'
val = np.ma.array([1, 2, 3, 4], mask=[False, False, True, False])
print(val)

value = ureg(from_units) * val 
value.ito(to_units)
print(value)