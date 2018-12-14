# Third-party
from sqlalchemy.types import TypeDecorator, REAL
from astropy.units import Quantity

__all__ = ['QuantityTypeClassFactory']

def QuantityTypeClassFactory(unit):
    class QuantityType(TypeDecorator):
        """ Custom type to handle `~astropy.units.Quantity` objects. """

        impl = REAL

        def process_bind_param(self, value, dialect):
            if isinstance(value, Quantity):
                return value.to(unit).value
            else:
                return value

        def process_result_value(self, value, dialect):
            if value is not None:
                value = value * unit
            return value

    QuantityType.__name__ = '{}QuantityType'.format(unit.physical_type.title().replace(' ', ''))
    return QuantityType
