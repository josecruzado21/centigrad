import math

class Variable:
    def __init__(self, value, parents = [], gradients = []):
        self.value = value
        self.parents = []
        self.gradients = [0]*len(self.parents) if len(self.parents)>0 else []

    def __repr__(self):
        return f"Variable(value={self.value})"
    
    def __add__(self, other_value):
        if (not isinstance(other_value, Variable)) & (type(other_value) in [int, float]):
            other_value = Variable(value = other_value)
        result = Variable(self.value + other_value.value)
        self.parents.append(result)
        other_value.parents.append(result)
        self.gradients.append(1)
        other_value.gradients.append(1)
        return result
    
    def __radd__(self, other_value):
        return self.__add__(other_value)
    
    def __mul__(self, other_value):
        if (not isinstance(other_value, Variable)) & (type(other_value) in [int, float]):
            other_value = Variable(value = other_value)
        result = Variable(self.value * other_value.value)
        self.parents.append(result)
        other_value.parents.append(result)
        self.gradients.append(other_value.value)
        other_value.gradients.append(self.value)
        return result
    
    def __rmul__(self, other_value):
        return self.__mul__(other_value)
    
    def __sub__(self, other_value):
        return self + (other_value*(-1))
    
    def __rsub__(self, other_value):
        return (self.__sub__(other_value))*(-1)
    
    def __pow__(self, other_value):
        result = Variable((self.value)**other_value)
        self.parents.append(result)
        self.gradients.append(other_value*(self.value**(other_value-1)))
        return result
    
    def __neg__(self):
        return (-1)*self
    
    def __truediv__(self, denominator):
        return self * (denominator)**(-1)
          
    def __rtruediv__(self, denominator):
        return (self.__truediv__(denominator))**(-1)
       
    def exp(self):
        result = Variable(math.exp(self.value))
        self.parents.append(result)
        self.gradients.append(math.exp(self.value))
        return result
    
    def sigmoid(self):
        result = (1) / (1 + (-self).exp())
        self.parents.append(result)
        self.gradients.append(((self.parents.value)*(1-self.parents.value)))
        return result
    
    def relu(self):
        result = Variable(max(0, self.value))
        self.parents.append(result)
        self.gradients.append(1 if (self.value > 0) else 0)
        return result
    
    def grad(self):
        if len(self.parents)>0:
            return sum([a * b for a, b in zip(self.gradients, [z.grad() for z in self.parents])])
        else:
            return 1