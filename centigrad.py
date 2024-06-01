class Value:
    def __init__(self, value, parent = None, gradient = 0):
        self.value = value
        self.parent = parent
        self.gradient = gradient if self.parent is not None else 1

    def __repr__(self):
        return f"Value(value={self.value})"
    
    def __add__(self, other_value):
        if (not isinstance(other_value, Value)) & (type(other_value) in [int, float]):
            other_value = Value(value = other_value)
        self.parent = Value(self.value + other_value.value)
        other_value.parent = self.parent
        self.gradient = 1
        other_value.gradient = 1
        return self.parent
    
    def __radd__(self, other_value):
        return self.__add__(other_value)
    
    def __mul__(self, other_value):
        if (not isinstance(other_value, Value)) & (type(other_value) in [int, float]):
            other_value = Value(value = other_value)
        self.parent = Value(self.value * other_value.value)
        other_value.parent = self.parent
        self.gradient = other_value.value
        other_value.gradient = self.value
        return self.parent
    
    def __rmul__(self, other_value):
        return self.__mul__(other_value)
    
    def __sub__(self, other_value):
        return self + (other_value*(-1))
    
    def __rsub__(self, other_value):
        return (self.__sub__(other_value))*(-1)
    
    def __pow__(self, other_value):
        self.parent = Value((self.value)**other_value)
        self.gradient = other_value*(self.value**(other_value-1))
        return self.parent
    
    def __neg__(self):
        return (-1)*self
    
    def __truediv__(self, denominator):
        return self * (denominator)**(-1)
          
    def __rtruediv__(self, denominator):
        return (self.__truediv__(denominator))**(-1)
       
    def exp(self):
        self.parent = Value(math.exp(self.value))
        self.gradient = math.exp(self.value)
        return self.parent
    
    def sigmoid(self):
        self.parent = (1) / (1 + (-self).exp())
        self.gradient = ((self.parent.value)*(1-self.parent.value))
        return self.parent
    
    def relu(self):
        self.parent = Value(max(0, self.value))
        self.gradient = 1 if (self.value > 0) else 0
        return self.parent
    
    def grad(self):
        if self.parent:
            return self.gradient*self.parent.grad()
        else:
            return 1