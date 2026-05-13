def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    
    for _ in range(steps):
        # f(x) = ax² + bx + c
        # f'(x) = 2ax + b
        gradient = 2 * a * x + b
        
        # Gradient descent update: x = x - lr * f'(x)
        x = x - lr * gradient
    
    return float(x)