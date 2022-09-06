def derivative(f, x, h=0.001):
    """
    Computes the derivative of function f using the finite approximation method.

    :param f: Function to be derived
    :param x: Independent variable to be passed to the function
    :param h: Finite difference(0.001 by default)
    :return: Value of the function derivative at X
    """
    if f(x) is None:
        return None

    if f(x + h) is None:
        return (f(x) - f(x - h)) / h
    if f(x - h) is None:
        return (f(x + h) - f(x)) / h

    return (f(x + h) - f(x - h)) / (2*h)
