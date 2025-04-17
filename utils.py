def print_model_analysis(model, m, k, n):
    """
    Print the analysis of computations and parameters for the model
    
    Args:
        model: The CNN model
        m: Number of filters in each convolutional layer
        k: Size of filters (k×k)
        n: Number of neurons in the dense layer
    """
    total_computations = model.calculate_computations(m, k, n)
    total_parameters = model.calculate_parameters(m, k, n)
    
    print("\n" + "="*50)
    print("MODEL ANALYSIS")
    print("="*50)
    print(f"Configuration: m={m} filters, k={k}×{k} filter size, n={n} dense neurons")
    print(f"Total computations: {total_computations:,}")
    print(f"Total parameters: {total_parameters:,}")
    
    print("\nSymbolic formula for computations:")
    print(f"Total computations = 150,528k²m + 16,660k²m² + 49mn + 10n")
    
    print("\nSymbolic formula for parameters:")
    print(f"Total parameters = 3k²m + 4k²m² + 49mn + 11n + 5m + 10")
    print("="*50)
